"""Quick LR sweep on expert 0 of layer 8 with real activations."""
import os
import json
import time
import copy
import gc
import numpy as np
import torch
from transformers import OlmoeForCausalLM

from src.quantize.serialize import load_quantized
from src.finetune.collect_activations import collect_layer_activations
from src.finetune.quant_expert import ft_one_expert

from src.quantize.serialize import dequant_target
from src.codes.ref import decode_hyb_batch
from safetensors import safe_open

LAYER = 8
EXPERT = 0
MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
QUANT_DIR = f"cache/quantized/L{LAYER:02d}"
LUT_PATH = "cache/codes/hyb_lut_init.npy"
DEVICE = "cuda:0"


def main():
    print("Loading model...")
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    rotary_emb = copy.deepcopy(model.model.rotary_emb).to(device=DEVICE, dtype=torch.float32)

    print(f"Collecting activations for layer {LAYER}...")
    captures = collect_layer_activations(
        model, rotary_emb, LAYER, cfg, DEVICE, budget_per_expert=1024,
    )
    X, Y = captures[EXPERT]

    # Free model — we don't need it anymore
    del model, rotary_emb, captures
    gc.collect()
    torch.cuda.empty_cache()

    # Load expert payloads once
    gate_p = load_quantized(f"{QUANT_DIR}/expert_{EXPERT:02d}_gate_proj.pt")
    up_p   = load_quantized(f"{QUANT_DIR}/expert_{EXPERT:02d}_up_proj.pt")
    down_p = load_quantized(f"{QUANT_DIR}/expert_{EXPERT:02d}_down_proj.pt")
    lut_init = np.load(LUT_PATH)

    X_dev = X.to(DEVICE)
    Y_dev = Y.to(DEVICE)

    print(f"\nX stats: shape={X.shape} mean={X.mean():.4f} std={X.std():.4f} max={X.abs().max():.4f}")
    print(f"Y stats: shape={Y.shape} mean={Y.mean():.4f} std={Y.std():.4f} max={Y.abs().max():.4f}")
    print(f"||Y||²/n = {(Y.float().pow(2).sum() / Y.numel()).item():.4e} (loss floor if W_q == 0)")

    for lr in [5e-4, 2e-3, 1e-2, 5e-2]:
        print(f"\n=== lr={lr} ===")
        t0 = time.time()
        result = ft_one_expert(
            gate_p, up_p, down_p, lut_init,
            X_dev, Y_dev, n_steps=500, lr=lr, device=DEVICE, verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  loss: {result['loss_init']:.4e} -> {result['loss_final']:.4e}  "
              f"({result['improvement']*100:.1f}%)  {elapsed:.0f}s")

    # What's the reconstruction error of the *current quantized weights* (no FT) on this data?
    # vs the fp16 forward (which is exactly Y by construction)?
    # loss_init = MSE(q_expert(X), fp16_expert(X)) should be SMALL relative to ||Y||²

    # Also check: what does PER-WEIGHT reconstruction error look like?


    decoder = lambda s: decode_hyb_batch(s, lut_init, Q=9)

    # Dequant gate_proj
    W_gate_q = dequant_target(gate_p, decoder)
    import json
    with open("cache/model/olmoe-1b-7b-0125/model.safetensors.index.json") as f:
        idx = json.load(f)
    key = "model.layers.8.mlp.experts.0.gate_proj.weight"
    shard = idx["weight_map"][key]
    with safe_open(f"cache/model/olmoe-1b-7b-0125/{shard}", framework="numpy") as f:
        W_gate_orig = f.get_tensor(key).astype(np.float32)

    abs_err = np.abs(W_gate_orig - W_gate_q).mean()
    rel_err = abs_err / np.abs(W_gate_orig).mean()
    sig_err_ratio = ((W_gate_orig - W_gate_q)**2).sum() / (W_gate_orig**2).sum()
    print(f"\nWeight space recon error (gate_proj):")
    print(f"  abs mean error: {abs_err:.4e}")
    print(f"  rel mean error: {rel_err*100:.1f}%")
    print(f"  ||W_q - W||² / ||W||² = {sig_err_ratio:.4f}")
    
if __name__ == "__main__":
    main()