"""Expert Hessian collector + monkey-patch for OlmoeExperts.forward.

The original OlmoeExperts.forward iterates over hit experts and calls
F.linear(current_state, gate_up_proj[i]) then F.linear(intermediate, down_proj[i]).
We patch it to mirror the computation while accumulating input outer products
into per-expert Hessian buffers.

The patch is installed per-instance (not on the class) so it's trivially
reversible and cannot leak between layers.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertHessianCollector:
    """Per-expert Hessians on GPU, fp32.

    For each of the 64 experts in a single layer, we track:
      gate_up_H[i]: hidden_size x hidden_size, input to gate_up_proj[i]
      down_H[i]:    intermediate x intermediate, input to down_proj[i]

    gate_up_H is shared between gate_proj and up_proj because they take
    identical inputs (gate_up_proj is a single fused matmul internally
    whose output is chunked into gate and up). When we quantize, we'll
    split the weight but reuse the same Hessian for both halves.
    """

    def __init__(self, num_experts, hidden_size, intermediate_size, device):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.device = device

        self.gate_up_H = torch.zeros(
            (num_experts, hidden_size, hidden_size),
            dtype=torch.float32, device=device,
        )
        self.down_H = torch.zeros(
            (num_experts, intermediate_size, intermediate_size),
            dtype=torch.float32, device=device,
        )
        self.gate_up_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
        self.down_counts = torch.zeros(num_experts, dtype=torch.long, device=device)

    def accumulate_gate_up(self, expert_idx, x):
        """x: (n_tokens_for_this_expert, hidden_size), fp32."""
        self.gate_up_H[expert_idx].addmm_(x.T, x)
        self.gate_up_counts[expert_idx] += x.shape[0]

    def accumulate_down(self, expert_idx, x):
        """x: (n_tokens_for_this_expert, intermediate_size), fp32."""
        self.down_H[expert_idx].addmm_(x.T, x)
        self.down_counts[expert_idx] += x.shape[0]

    def save(self, out_dir, layer_idx):
        """Normalize by token count, symmetrize, save one file per (expert, proj)."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        for e in range(self.num_experts):
            n_gu = int(self.gate_up_counts[e])
            n_dn = int(self.down_counts[e])

            H_gu = self.gate_up_H[e] / max(n_gu, 1)
            H_gu = 0.5 * (H_gu + H_gu.T)
            torch.save({
                "H": H_gu.cpu(),
                "n_tokens": n_gu,
                "layer": layer_idx,
                "expert": e,
                "kind": "expert",
                "proj": "gate_up",  # shared by gate_proj and up_proj
            }, os.path.join(out_dir, f"expert_{e:02d}_gate_up.pt"))

            H_dn = self.down_H[e] / max(n_dn, 1)
            H_dn = 0.5 * (H_dn + H_dn.T)
            torch.save({
                "H": H_dn.cpu(),
                "n_tokens": n_dn,
                "layer": layer_idx,
                "expert": e,
                "kind": "expert",
                "proj": "down",
            }, os.path.join(out_dir, f"expert_{e:02d}_down.pt"))


def _patched_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Mirror of OlmoeExperts.forward with per-expert input capture.

    Must match the original computation EXACTLY (verified by the tripwire's
    bit-identity check on the layer output). The only difference is the
    accumulate_* calls into self._hessian_collector.
    """
    collector = self._hessian_collector
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # === gate_up input capture ===
        ei = int(expert_idx)
        collector.accumulate_gate_up(ei, current_state.to(torch.float32))

        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        mlp_intermediate = self.act_fn(gate) * up

        # === down input capture ===
        collector.accumulate_down(ei, mlp_intermediate.to(torch.float32))

        current_hidden_states = F.linear(mlp_intermediate, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


def install_expert_patch(experts_module, collector):
    """Install the patched forward on a single OlmoeExperts instance.

    Returns a restore function; call it to undo the patch.
    """
    if hasattr(experts_module, "_original_forward"):
        raise RuntimeError("Already patched; restore first before re-installing.")
    experts_module._original_forward = experts_module.forward
    experts_module._hessian_collector = collector
    experts_module.forward = types.MethodType(_patched_experts_forward, experts_module)

    def restore():
        experts_module.forward = experts_module._original_forward
        del experts_module._original_forward
        del experts_module._hessian_collector
    return restore