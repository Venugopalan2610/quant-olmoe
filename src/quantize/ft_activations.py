"""Per-layer streaming activation collector for fine-tuning.

For one layer, runs fp16 calibration tokens through and captures:
  - per-expert (X_e, Y_e): subsampled inputs and outputs
  - attention block (X_attn, Y_attn): subsampled inputs and outputs
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def collect_layer_activations(model, layer_idx, calib_tokens,
                                tokens_per_expert=1024, num_experts=64,
                                device="cuda:0"):
    """Run calibration tokens through one layer of fp16 model, capture activations.

    Args:
        model: fp16 OLMoE model on device (must have layer_idx loaded)
        calib_tokens: (n_tokens, hidden_size) fp16 tensor on device
        tokens_per_expert: subsample to this many per expert (1024 default)
    Returns:
        dict with:
            'expert_inputs':  list of 64 tensors (tokens_per_expert, 2048)
            'expert_outputs': list of 64 tensors (tokens_per_expert, 2048)
            'attn_input':     (tokens_per_expert, 2048)
            'attn_output':    (tokens_per_expert, 2048)
            'attn_position_ids': (tokens_per_expert,) for RoPE
    """
    layer = model.model.layers[layer_idx]

    # Hook to capture attention input/output
    attn_io = {}
    def attn_hook(module, inp, out):
        attn_io['input'] = inp[0].detach()
        attn_io['output'] = out[0].detach() if isinstance(out, tuple) else out.detach()
    h1 = layer.self_attn.register_forward_hook(attn_hook)

    # Hooks to capture per-expert inputs/outputs and routing
    expert_io = {e: {'inputs': [], 'outputs': []} for e in range(num_experts)}

    def make_expert_hook(e):
        def hook(module, inp, out):
            expert_io[e]['inputs'].append(inp[0].detach())
            expert_io[e]['outputs'].append(out.detach())
        return hook

    expert_hooks = []
    for e in range(num_experts):
        h = layer.mlp.experts[e].register_forward_hook(make_expert_hook(e))
        expert_hooks.append(h)

    # Run forward pass
    with torch.no_grad():
        # We need a way to feed activations into ONE layer. Easiest:
        # run the full model forward on calibration tokens, layer_idx's hooks fire.
        # Reuse the same calibration loop you used for Hessian collection.
        ...  # TODO: lift the run-calibration-through-model loop from src/calibration/run.py

    # Cleanup hooks
    h1.remove()
    for h in expert_hooks:
        h.remove()

    # Subsample per expert
    expert_inputs = []
    expert_outputs = []
    for e in range(num_experts):
        if not expert_io[e]['inputs']:
            # Rare expert with zero tokens
            expert_inputs.append(None)
            expert_outputs.append(None)
            continue
        x_cat = torch.cat(expert_io[e]['inputs'], dim=0)  # (n_tokens_e, 2048)
        y_cat = torch.cat(expert_io[e]['outputs'], dim=0)
        n = x_cat.shape[0]
        if n > tokens_per_expert:
            idx = torch.randperm(n, device=device)[:tokens_per_expert]
            x_cat = x_cat[idx]
            y_cat = y_cat[idx]
        expert_inputs.append(x_cat.float())
        expert_outputs.append(y_cat.float())

    # Subsample attention
    n_attn = attn_io['input'].shape[0] * attn_io['input'].shape[1] if attn_io['input'].ndim == 3 else attn_io['input'].shape[0]
    # Flatten if (batch, seq, hidden) → (batch*seq, hidden)
    x_attn = attn_io['input'].reshape(-1, attn_io['input'].shape[-1])
    y_attn = attn_io['output'].reshape(-1, attn_io['output'].shape[-1])
    if x_attn.shape[0] > tokens_per_expert:
        idx = torch.randperm(x_attn.shape[0], device=device)[:tokens_per_expert]
        x_attn = x_attn[idx]
        y_attn = y_attn[idx]

    return {
        'expert_inputs': expert_inputs,
        'expert_outputs': expert_outputs,
        'attn_input': x_attn.float(),
        'attn_output': y_attn.float(),
    }