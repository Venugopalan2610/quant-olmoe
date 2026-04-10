"""Per-expert joint LUT fine-tuning.

For one expert, jointly optimize gate/up/down LUTs to minimize
MSE(quantized_expert(X), fp16_expert(X)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.quantize.lut_ft import precompute_walk_states, differentiable_dequant


class QuantizedExpert(nn.Module):
    """Differentiable quantized expert. LUTs are the only trainable params."""

    def __init__(self, gate_payload, up_payload, down_payload, lut_init,
                 device="cuda:0"):
        super().__init__()
        self.device = device

        # Trainable LUTs (one per projection, initialized from global LUT)
        self.lut_gate = nn.Parameter(torch.from_numpy(lut_init).float().to(device))
        self.lut_up   = nn.Parameter(torch.from_numpy(lut_init).float().to(device))
        self.lut_down = nn.Parameter(torch.from_numpy(lut_init).float().to(device))

        # Frozen pieces (precompute walks once)
        self.gate_walks = self._make_walks(gate_payload)
        self.up_walks   = self._make_walks(up_payload)
        self.down_walks = self._make_walks(down_payload)

        self.gate_meta = self._extract_meta(gate_payload)
        self.up_meta   = self._extract_meta(up_payload)
        self.down_meta = self._extract_meta(down_payload)

    def _make_walks(self, p):
        walks = precompute_walk_states(p["bitstreams"], p["start_states"])
        return torch.from_numpy(walks).long().to(self.device)

    def _extract_meta(self, p):
        return {
            "sign_l": torch.from_numpy(p["sign_l"].astype(np.float32)).to(self.device),
            "sign_r": torch.from_numpy(p["sign_r"].astype(np.float32)).to(self.device),
            "W_scale": float(p["W_scale"]),
            "shape": tuple(p["shape"]),
        }

    def _dequant(self, walks, lut, meta):
        return differentiable_dequant(
            walks, lut, meta["sign_l"], meta["sign_r"],
            meta["W_scale"], meta["shape"]
        )

    def forward(self, x):
        # x: (batch, hidden_in)
        W_gate = self._dequant(self.gate_walks, self.lut_gate, self.gate_meta)
        W_up   = self._dequant(self.up_walks,   self.lut_up,   self.up_meta)
        W_down = self._dequant(self.down_walks, self.lut_down, self.down_meta)

        gate_out = F.silu(x @ W_gate.T)
        up_out   = x @ W_up.T
        hidden   = gate_out * up_out
        return hidden @ W_down.T


def ft_expert(gate_payload, up_payload, down_payload, lut_init,
               X, Y, n_steps=500, lr=5e-4, device="cuda:0", verbose=False):
    """Fine-tune one expert's LUTs jointly.

    Args:
        gate/up/down_payload: dicts from load_quantized
        lut_init: (512, 2) numpy
        X: (n_tokens, hidden_in) torch.float32 on device — expert input
        Y: (n_tokens, hidden_out) torch.float32 on device — fp16 expert output
    Returns:
        dict with loss_init, loss_final, luts (3-tuple of np arrays)
    """
    expert = QuantizedExpert(gate_payload, up_payload, down_payload, lut_init, device)
    optimizer = torch.optim.Adam(expert.parameters(), lr=lr)

    with torch.no_grad():
        Y_pred = expert(X)
        loss_init = F.mse_loss(Y_pred, Y).item()

    for step in range(n_steps):
        optimizer.zero_grad()
        Y_pred = expert(X)
        loss = F.mse_loss(Y_pred, Y)
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f"      step {step}: loss={loss.item():.4e}")

    loss_final = loss.item()
    return {
        "loss_init": loss_init,
        "loss_final": loss_final,
        "lut_gate": expert.lut_gate.detach().cpu().numpy(),
        "lut_up":   expert.lut_up.detach().cpu().numpy(),
        "lut_down": expert.lut_down.detach().cpu().numpy(),
    }