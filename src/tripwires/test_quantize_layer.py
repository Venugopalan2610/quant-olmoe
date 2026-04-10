"""Tripwire A.8.2: single-layer quantization driver.

Runs quantize_layer(8) end-to-end. Verifies:
A.8.2.1: All 132 files produced
A.8.2.2: Dequant round-trip on 4 random targets matches blockldlq direct output
A.8.2.3: Per-target proxy loss distribution sane
A.8.2.4: Wall clock projects to feasible full-model run

Run: python -m src.tripwires.test_quantize_layer
"""
import os
import sys
import time
import json
import numpy as np
import torch
from safetensors import safe_open

from src.codes.ref import decode_hyb_batch
from src.rht.transform import make_sign_vector
from src.quantize.blockldlq import blockldlq
from src.quantize.serialize import load_quantized, dequant_target
from src.quantize.quantize_layer import quantize_layer

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HESSIAN_DIR = "cache/hessians"
QUANTIZED_DIR = "cache/quantized"
LUT_PATH = "cache/codes/hyb_lut_init.npy"


def _decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def main():
    print("=" * 60)
    print("Tripwire A.8.2: single-layer quantization driver (L8)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    # Clean any stale L8 quantized output
    out_dir = os.path.join(QUANTIZED_DIR, "L08")
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)

    print("\nRunning quantize_layer(8)...")
    t0 = time.time()
    results = quantize_layer(layer_idx=8, verbose=True)
    layer_time = time.time() - t0

    # A.8.2.1: file count
    print("\nA.8.2.1: file count")
    print("-" * 60)
    files = sorted(os.listdir(out_dir))
    print(f"  files in {out_dir}: {len(files)}")
    expected = 4 + 64 * 3  # 4 attn + 64 experts * 3 projs
    files_ok = len(files) == expected
    print(f"  [{'PASS' if files_ok else 'FAIL'}] file count {len(files)} == expected {expected}")

    # A.8.2.2: dequant round-trip on 4 random targets
    print("\nA.8.2.2: dequant round-trip")
    print("-" * 60)
    decode = _decoder()

    # Pick 4 targets to spot-check
    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(results), size=4, replace=False)

    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        idx_json = json.load(f)

    roundtrip_ok = True
    for i in sample_idxs:
        r = results[i]
        target_path = os.path.join(out_dir, f"{r['name']}.pt")
        saved = load_quantized(target_path)

        # Re-quantize the same target to compare
        if r["kind"] == "attention":
            key = f"model.layers.8.self_attn.{r['proj']}.weight"
            H_path = os.path.join(HESSIAN_DIR, "L08", f"attn_{r['proj']}.pt")
        else:
            key = f"model.layers.8.mlp.experts.{r['expert']}.{r['proj']}.weight"
            h_kind = "down" if r["proj"] == "down_proj" else "gate_up"
            H_path = os.path.join(HESSIAN_DIR, "L08", f"expert_{r['expert']:02d}_{h_kind}.pt")

        shard = idx_json["weight_map"][key]
        with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
            W = f.get_tensor(key).astype(np.float32)
        H = torch.load(H_path, weights_only=True)["H"].numpy()

        sign_l = saved["sign_l"]
        sign_r = saved["sign_r"]

        Wh_direct, _, _ = blockldlq(W, H, sign_l, sign_r, decode, use_cuda=True)
        Wh_dequant = dequant_target(saved, decode)

        max_diff = float(np.abs(Wh_direct - Wh_dequant).max())
        ok = max_diff < 1e-12
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {r['name']}: max diff = {max_diff:.2e}")
        if not ok:
            roundtrip_ok = False

    # A.8.2.3: proxy loss distribution
    print("\nA.8.2.3: proxy loss distribution")
    print("-" * 60)
    attn_losses = [r["proxy_loss"] for r in results if r["kind"] == "attention"]
    expert_losses = [r["proxy_loss"] for r in results if r["kind"] == "expert"]

    print(f"  attention proxy losses (n={len(attn_losses)}):")
    print(f"    min={min(attn_losses):.3e} max={max(attn_losses):.3e} "
          f"mean={np.mean(attn_losses):.3e}")
    print(f"  expert proxy losses (n={len(expert_losses)}):")
    print(f"    min={min(expert_losses):.3e} max={max(expert_losses):.3e} "
          f"mean={np.mean(expert_losses):.3e}")

    no_extreme_outliers = max(expert_losses) < 100 * np.median(expert_losses)
    print(f"  [{'PASS' if no_extreme_outliers else 'FAIL'}] "
          f"no extreme outliers (max < 100x median)")

    # A.8.2.4: wall clock
    print("\nA.8.2.4: wall clock")
    print("-" * 60)
    avg_per_target = layer_time / len(results)
    full_model_hours = avg_per_target * 132 * 16 / 3600
    print(f"  L8 total: {layer_time:.0f}s ({layer_time/60:.1f} min)")
    print(f"  per-target avg: {avg_per_target:.1f}s")
    print(f"  full model (16 layers x 132 targets): "
          f"{full_model_hours:.1f} hours")
    feasible = full_model_hours < 24
    print(f"  [{'PASS' if feasible else 'FAIL'}] full model < 24 hours")

    print("\n" + "=" * 60)
    all_ok = files_ok and roundtrip_ok and no_extreme_outliers and feasible
    if all_ok:
        print("A.8.2 GATE: PASS — single-layer driver verified.")
        print("Ready for A.8.3 (full-model run).")
        sys.exit(0)
    else:
        print("A.8.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()