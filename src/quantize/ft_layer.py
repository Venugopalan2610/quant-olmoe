"""Driver: FT all targets in one layer.

For each target in cache/quantized/L{NN}/, load bitstreams + W + H,
run ft_one_target, save fine-tuned LUT to cache/quantized_ft/L{NN}/.
"""
import os
import json
import time
import numpy as np
import torch
from safetensors import safe_open

from src.quantize.serialize import load_quantized
from src.quantize.lut_ft import ft_one_target

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HESSIAN_DIR = "cache/hessians"
QUANTIZED_DIR = "cache/quantized"
QUANTIZED_FT_DIR = "cache/quantized_ft"
LUT_INIT_PATH = "cache/codes/hyb_lut_init.npy"


def ft_layer(layer_idx, num_experts=64, n_steps=500, lr=5e-4, verbose=True):
    in_dir = os.path.join(QUANTIZED_DIR, f"L{layer_idx:02d}")
    out_dir = os.path.join(QUANTIZED_FT_DIR, f"L{layer_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)
    
    lut_init = np.load(LUT_INIT_PATH)  # (512, 2)
    layer_luts = {}  # target_name → tuned lut
    layer_stats = []
    
    layer_start = time.time()
    
    # Iterate over all targets in the layer (132 of them)
    target_files = sorted(os.listdir(in_dir))
    for target_file in target_files:
        if not target_file.endswith(".pt"):
            continue
        target_name = target_file[:-3]
        
        # Load bitstream payload
        bs_dict = load_quantized(os.path.join(in_dir, target_file))
        
        # Load reference W from safetensors
        W = _load_weight_for_target(target_name, layer_idx)
        
        # Load Hessian
        H = _load_hessian_for_target(target_name, layer_idx)
        
        # Run FT
        t0 = time.time()
        result = ft_one_target(W, H, bs_dict, lut_init, n_steps=n_steps, lr=lr)
        dt = time.time() - t0
        
        layer_luts[target_name] = result["lut_final"]
        layer_stats.append({
            "name": target_name,
            "loss_init": result["loss_init"],
            "loss_final": result["loss_final"],
            "improvement": (result["loss_init"] - result["loss_final"]) / result["loss_init"],
            "wall_time": dt,
        })
        
        if verbose and len(layer_stats) % 16 == 0:
            elapsed = time.time() - layer_start
            avg_improvement = np.mean([s["improvement"] for s in layer_stats])
            print(f"  L{layer_idx:02d} progress: {len(layer_stats)}/{len(target_files)}, "
                  f"avg improvement {avg_improvement*100:.1f}%, elapsed {elapsed:.0f}s")
    
    # Save all LUTs for this layer in one file
    torch.save(layer_luts, os.path.join(out_dir, "luts.pt"))
    
    # Save stats
    with open(os.path.join(out_dir, "ft_stats.json"), "w") as f:
        json.dump(layer_stats, f, indent=2)
    
    layer_total = time.time() - layer_start
    avg_improvement = np.mean([s["improvement"] for s in layer_stats])
    print(f"  L{layer_idx:02d} FT done: {len(layer_stats)} targets in {layer_total:.0f}s, "
          f"avg loss reduction {avg_improvement*100:.1f}%")
    
    return layer_stats


def _load_weight_for_target(target_name, layer_idx):
    """Map target name like 'expert_03_gate_proj' to safetensors key, load tensor."""
    # ...lift this from quantize_layer.py
    raise NotImplementedError


def _load_hessian_for_target(target_name, layer_idx):
    """Map target name to Hessian file path, load."""
    raise NotImplementedError