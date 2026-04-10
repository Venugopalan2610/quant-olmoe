"""OLMoE model adapter — handles fused expert storage in modern transformers.

OlmoeExperts stores all 64 experts as stacked 3D tensors:
  gate_up_proj: (num_experts, 2*intermediate, hidden)
  down_proj:    (num_experts, hidden, intermediate)

The first half of gate_up_proj rows is the gate projection; the second half
is the up projection. We quantize gate and up separately (matching QTIP's
"no matrix fusion" choice in paper appendix A.3.5).

Each QuantTarget exposes get_weight()/set_weight() that handles the slice
view onto the fused tensor.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterator, Optional, Callable


def get_arch_config(model):
    cfg = model.config
    return {
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
    }


@dataclass
class QuantTarget:
    """One quantizable weight matrix.

    name: human-readable ID, e.g. 'L00.attn.q_proj' or 'L00.E05.gate_proj'
    kind: 'attention' or 'expert'
    layer_idx: int
    expert_idx: int or None
    proj: 'q_proj' | 'k_proj' | 'v_proj' | 'o_proj' | 'gate_proj' | 'up_proj' | 'down_proj'
    in_features: int
    out_features: int
    get_weight: callable returning a fresh tensor (copy, safe to mutate)
    set_weight: callable taking a tensor and writing it back to the model
    """
    name: str
    kind: str
    layer_idx: int
    expert_idx: Optional[int]
    proj: str
    in_features: int
    out_features: int
    get_weight: Callable[[], torch.Tensor]
    set_weight: Callable[[torch.Tensor], None]


def _make_attn_target(model, layer_idx, proj):
    """Build a QuantTarget for an attention projection (a real nn.Linear)."""
    layer = model.model.layers[layer_idx]
    linear = getattr(layer.self_attn, proj)

    def get():
        return linear.weight.detach().clone()

    def set_(w):
        with torch.no_grad():
            linear.weight.copy_(w.to(linear.weight.dtype).to(linear.weight.device))

    return QuantTarget(
        name=f"L{layer_idx:02d}.attn.{proj}",
        kind="attention",
        layer_idx=layer_idx,
        expert_idx=None,
        proj=proj,
        in_features=linear.in_features,
        out_features=linear.out_features,
        get_weight=get,
        set_weight=set_,
    )


def _make_expert_target(model, layer_idx, expert_idx, proj):
    """Build a QuantTarget for an expert projection (a slice of fused storage).

    For gate_proj/up_proj: a slice of gate_up_proj[expert_idx].
      gate_up_proj has shape (E, 2*I, H). gate is rows [0:I], up is rows [I:2I].
      Each "logical Linear" has shape (I, H) used as F.linear weight.
    For down_proj: gate_up_proj[expert_idx] is shape (H, I).
    """
    experts = model.model.layers[layer_idx].mlp.experts
    cfg = model.config
    H = cfg.hidden_size
    I = cfg.intermediate_size

    if proj in ("gate_proj", "up_proj"):
        offset = 0 if proj == "gate_proj" else I
        out_features = I
        in_features = H

        def get():
            return experts.gate_up_proj[expert_idx, offset:offset + I, :].detach().clone()

        def set_(w):
            with torch.no_grad():
                experts.gate_up_proj[expert_idx, offset:offset + I, :].copy_(
                    w.to(experts.gate_up_proj.dtype).to(experts.gate_up_proj.device)
                )
    elif proj == "down_proj":
        out_features = H
        in_features = I

        def get():
            return experts.down_proj[expert_idx].detach().clone()

        def set_(w):
            with torch.no_grad():
                experts.down_proj[expert_idx].copy_(
                    w.to(experts.down_proj.dtype).to(experts.down_proj.device)
                )
    else:
        raise ValueError(f"Unknown expert proj: {proj}")

    return QuantTarget(
        name=f"L{layer_idx:02d}.E{expert_idx:02d}.{proj}",
        kind="expert",
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        proj=proj,
        in_features=in_features,
        out_features=out_features,
        get_weight=get,
        set_weight=set_,
    )


def enumerate_quant_targets(model) -> Iterator[QuantTarget]:
    """Yield all quantizable weight matrices in the model.

    Order: layer-major, with attention before experts within each layer.
    """
    cfg = get_arch_config(model)
    for layer_idx in range(cfg["num_hidden_layers"]):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield _make_attn_target(model, layer_idx, proj)
        for expert_idx in range(cfg["num_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield _make_expert_target(model, layer_idx, expert_idx, proj)


def discover_layer_structure(model, layer_idx=0):
    """Print structural info about one layer for inspection."""
    layer = model.model.layers[layer_idx]
    print(f"\nLayer {layer_idx} structure:")
    print("=" * 60)
    for name, module in layer.named_modules():
        if name == "":
            continue
        cls = type(module).__name__
        if isinstance(module, nn.Linear):
            print(f"  {name}: {cls}  weight={tuple(module.weight.shape)}")
        elif "Moe" in cls or "Expert" in cls or "Router" in cls:
            print(f"  {name}: {cls}")

    # Also show fused expert tensors
    experts = layer.mlp.experts
    print(f"  mlp.experts.gate_up_proj: shape={tuple(experts.gate_up_proj.shape)}")
    print(f"  mlp.experts.down_proj:    shape={tuple(experts.down_proj.shape)}")
    print("=" * 60)