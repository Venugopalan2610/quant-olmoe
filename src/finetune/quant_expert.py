"""Differentiable quantized expert for joint LUT fine-tuning.

Wraps three quantized projections (gate, up, down) into a torch nn.Module
where the only trainable parameters are the three LUTs. Forward is the
exact OLMoE expert MLP: down(SiLU(gate(x)) * up(x)).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.lut_ft import precompute_walk_states, differentiable_dequant


class QuantizedExpert(nn.Module):
    """Joint differentiable expert. Trainable: 3 LUTs. Frozen: trellis walks, signs, scales."""

    def __init__(self, gate_payload, up_payload, down_payload, lut_init,
                 device="cuda:0", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Trainable LUTs (one per projection, each initialized from global LUT)
        self.lut_gate = nn.Parameter(
            torch.from_numpy(lut_init.copy()).to(device, dtype)
        )
        self.lut_up = nn.Parameter(
            torch.from_numpy(lut_init.copy()).to(device, dtype)
        )
        self.lut_down = nn.Parameter(
            torch.from_numpy(lut_init.copy()).to(device, dtype)
        )

        # Frozen pieces — precompute walks once and store on device
        self._setup_target("gate", gate_payload)
        self._setup_target("up", up_payload)
        self._setup_target("down", down_payload)

    def _setup_target(self, name, payload):
        cfg = payload["config"]
        L_bits = cfg["L"]
        kV = cfg["k"] * cfg["V"]

        walks = precompute_walk_states(
            payload["bitstreams"], payload["start_states"],
            L_bits=L_bits, kV=kV,
        )

        self.register_buffer(
            f"{name}_walks",
            torch.from_numpy(walks).long().to(self.device),
        )
        self.register_buffer(
            f"{name}_sign_l",
            torch.from_numpy(payload["sign_l"].astype(np.float32)).to(self.device, self.dtype),
        )
        self.register_buffer(
            f"{name}_sign_r",
            torch.from_numpy(payload["sign_r"].astype(np.float32)).to(self.device, self.dtype),
        )

        # Store as plain attributes (not buffers — they're scalars/tuples)
        setattr(self, f"{name}_W_scale", float(payload["W_scale"]))
        setattr(self, f"{name}_shape", tuple(payload["shape"]))
        setattr(self, f"{name}_Tx", cfg["Tx"])
        setattr(self, f"{name}_Ty", cfg["Ty"])
        setattr(self, f"{name}_V", cfg["V"])
        setattr(self, f"{name}_L_bits", L_bits)

    def _dequant(self, name, lut):
        walks = getattr(self, f"{name}_walks")
        sign_l = getattr(self, f"{name}_sign_l")
        sign_r = getattr(self, f"{name}_sign_r")
        return differentiable_dequant(
            walks, lut, sign_l, sign_r,
            W_scale=getattr(self, f"{name}_W_scale"),
            shape=getattr(self, f"{name}_shape"),
            Tx=getattr(self, f"{name}_Tx"),
            Ty=getattr(self, f"{name}_Ty"),
            V=getattr(self, f"{name}_V"),
            Q=9,
            L_bits=getattr(self, f"{name}_L_bits"),
        )

    def materialize_weights(self):
        """Return current (W_gate, W_up, W_down) from LUTs. Differentiable."""
        W_gate = self._dequant("gate", self.lut_gate)
        W_up   = self._dequant("up",   self.lut_up)
        W_down = self._dequant("down", self.lut_down)
        return W_gate, W_up, W_down

    def forward(self, x):
        """OLMoE expert forward: down(SiLU(gate(x)) * up(x)).

        x: (batch, hidden_in) torch.float32
        Returns: (batch, hidden_in) torch.float32
        """
        W_gate, W_up, W_down = self.materialize_weights()
        # OLMoE: gate_proj and up_proj take input of dim hidden_size,
        # output dim intermediate_size. So weights are (intermediate, hidden).
        # F.linear(x, W) computes x @ W.T → so x @ W_gate.T gives (batch, intermediate).
        gate_out = F.silu(F.linear(x, W_gate))
        up_out = F.linear(x, W_up)
        hidden = gate_out * up_out
        return F.linear(hidden, W_down)


def ft_one_expert(gate_payload, up_payload, down_payload, lut_init,
                   X, Y, n_steps=500, lr=5e-4, device="cuda:0", verbose=False):
    """Fine-tune one expert's three LUTs jointly via forward reconstruction.

    Args:
        gate/up/down_payload: dicts from load_quantized
        lut_init: (512, 2) numpy float32, initial LUT shared by all three
        X: (n_tokens, hidden_size) torch.float32 on device — expert input (fp32)
        Y: (n_tokens, hidden_size) torch.float32 on device — fp16 expert output (fp32)
    Returns:
        dict with loss_init, loss_final, lut_gate, lut_up, lut_down (numpy arrays)
    """
    expert = QuantizedExpert(gate_payload, up_payload, down_payload, lut_init, device=device)
    optimizer = torch.optim.Adam(
        [expert.lut_gate, expert.lut_up, expert.lut_down],
        lr=lr,
    )

    with torch.no_grad():
        Y_pred = expert(X)
        loss_init = F.mse_loss(Y_pred, Y).item()

    for step in range(n_steps):
        optimizer.zero_grad()
        Y_pred = expert(X)
        loss = F.mse_loss(Y_pred, Y)
        loss.backward()
        optimizer.step()

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"      step {step:4d}: loss={loss.item():.6e}")

    loss_final = loss.item()

    return {
        "loss_init": loss_init,
        "loss_final": loss_final,
        "improvement": (loss_init - loss_final) / loss_init,
        "lut_gate": expert.lut_gate.detach().cpu().numpy(),
        "lut_up":   expert.lut_up.detach().cpu().numpy(),
        "lut_down": expert.lut_down.detach().cpu().numpy(),
    }
    
# ============================================================================
# H-weighted FT for linear projections (attention q/k/v/o)
# ============================================================================

class QuantizedLinear(nn.Module):
    """Single quantized linear projection with trainable LUT."""

    def __init__(self, payload, lut_init, device="cuda:0", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        cfg = payload["config"]
        L_bits = cfg["L"]
        kV = cfg["k"] * cfg["V"]

        self.lut = nn.Parameter(torch.from_numpy(lut_init.copy()).to(device, dtype))

        walks = precompute_walk_states(
            payload["bitstreams"], payload["start_states"],
            L_bits=L_bits, kV=kV,
        )
        self.register_buffer("walks", torch.from_numpy(walks).long().to(device))
        self.register_buffer(
            "sign_l", torch.from_numpy(payload["sign_l"].astype(np.float32)).to(device, dtype)
        )
        self.register_buffer(
            "sign_r", torch.from_numpy(payload["sign_r"].astype(np.float32)).to(device, dtype)
        )
        self.W_scale = float(payload["W_scale"])
        self.shape = tuple(payload["shape"])
        self.Tx = cfg["Tx"]; self.Ty = cfg["Ty"]; self.V = cfg["V"]; self.L_bits = L_bits

    def materialize(self):
        return differentiable_dequant(
            self.walks, self.lut, self.sign_l, self.sign_r,
            W_scale=self.W_scale, shape=self.shape,
            Tx=self.Tx, Ty=self.Ty, V=self.V, Q=9, L_bits=self.L_bits,
        )


def ft_one_linear_hweighted(payload, W_ref, H, lut_init,
                              n_steps=500, lr=5e-4, damp=0.01,
                              device="cuda:0", verbose=False):
    """H-weighted weight reconstruction FT for one linear projection.

    Loss: ||(W_ref - W_q) @ L_chol||²_F  where  L_chol = chol(H + damp*I)

    This is mathematically equivalent in expectation to the per-token MSE
    on the projection's output activations, given calibration distribution H.

    Args:
        payload: dict from load_quantized
        W_ref: (m, n) numpy float32 — the original fp16 weight (cast to fp32)
        H: (n, n) numpy float32 — input Hessian, normalized
        lut_init: (512, 2) numpy
    Returns:
        dict with loss_init, loss_final, lut_final
    """
    # Damped Cholesky factor
    n = H.shape[0]
    H_d = H.astype(np.float64) + damp * np.trace(H) / n * np.eye(n)
    L_chol_np = np.linalg.cholesky(H_d).astype(np.float32)

    L_chol = torch.from_numpy(L_chol_np).to(device)
    W_t = torch.from_numpy(W_ref.astype(np.float32)).to(device)

    qlin = QuantizedLinear(payload, lut_init, device=device)
    optimizer = torch.optim.Adam([qlin.lut], lr=lr)

    with torch.no_grad():
        W_q = qlin.materialize()
        residual = (W_t - W_q) @ L_chol
        loss_init = residual.pow(2).sum().item() / W_t.numel()

    for step in range(n_steps):
        optimizer.zero_grad()
        W_q = qlin.materialize()
        residual = (W_t - W_q) @ L_chol
        loss = residual.pow(2).sum() / W_t.numel()
        loss.backward()
        optimizer.step()

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"      step {step:4d}: loss={loss.item():.6e}")

    return {
        "loss_init": loss_init,
        "loss_final": loss.item(),
        "improvement": (loss_init - loss.item()) / loss_init,
        "lut_final": qlin.lut.detach().cpu().numpy(),
    }