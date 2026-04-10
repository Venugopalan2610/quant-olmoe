"""Tripwire: FT one target, verify loss decreases meaningfully."""
import numpy as np
import torch
from src.quantize.serialize import load_quantized
from src.quantize.lut_ft import ft_one_target

# Pick L08 expert_00 gate_proj as canonical test target
TARGET_FILE = "cache/quantized/L08/expert_00_gate_proj.pt"
W_KEY = "model.layers.8.mlp.experts.0.gate_proj.weight"
H_FILE = "cache/hessians/L08/expert_00_gate_up.pt"

def main():
    # Load W
    from safetensors import safe_open
    import json
    with open("cache/model/olmoe-1b-7b-0125/model.safetensors.index.json") as f:
        idx = json.load(f)
    shard = idx["weight_map"][W_KEY]
    with safe_open(f"cache/model/olmoe-1b-7b-0125/{shard}", framework="numpy") as f:
        W = f.get_tensor(W_KEY).astype(np.float32)
    
    # Load H
    H = torch.load(H_FILE, weights_only=True)["H"].numpy()
    
    # Load bitstream
    bs_dict = load_quantized(TARGET_FILE)
    
    # Load init LUT
    lut_init = np.load("cache/codes/hyb_lut_init.npy")
    
    # Run FT
    print(f"FT one target: shape={W.shape}, n_steps=500")
    result = ft_one_target(W, H, bs_dict, lut_init, n_steps=500, lr=5e-4, verbose=True)
    
    print(f"\nLoss init:  {result['loss_init']:.4e}")
    print(f"Loss final: {result['loss_final']:.4e}")
    improvement = (result['loss_init'] - result['loss_final']) / result['loss_init']
    print(f"Improvement: {improvement*100:.1f}%")
    
    # Pass criteria: loss should drop by at least 5%
    if improvement > 0.05:
        print("[PASS] Loss reduced >5%")
    else:
        print("[FAIL] Loss reduction <5% — check gradient flow")

if __name__ == "__main__":
    main()