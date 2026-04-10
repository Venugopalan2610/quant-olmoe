"""Tripwire A.8.1: quantized format save/load round-trip.

A.8.1.1: Synthetic 128x128 — blockldlq(...)[0] == dequant(save(blockldlq returns))
A.8.1.2: Real L8 E0 down_proj — same round-trip on actual OLMoE data
A.8.1.3: File size sanity check

Run: python -m src.tripwires.test_quant_serialize
"""
import os
import sys
import json
import numpy as np
import torch
from safetensors import safe_open

from src.codes.ref import decode_hyb_batch
from src.rht.transform import make_sign_vector
from src.quantize.blockldlq import blockldlq
from src.quantize.serialize import save_quantized, load_quantized, dequant_target


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HESSIAN_DIR = "cache/hessians"
LUT_PATH = "cache/codes/hyb_lut_init.npy"
TMP_DIR = "/tmp/qtip_olmoe_a81"


def _decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def _round_trip(W, H, sign_l, sign_r, label):
    decode = _decoder()
    print(f"\n  --- {label} ---")
    print(f"  W shape: {W.shape}")

    # Quantize with bitstream return
    Wh_direct, Wh_tilde, proxy, bs_dict = blockldlq(
        W, H, sign_l, sign_r, decode, use_cuda=True,
        return_bitstreams=True,
    )

    # Save
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, f"{label}.pt")
    save_quantized(
        out_path,
        bitstreams=bs_dict["bitstreams"],
        start_states=bs_dict["start_states"],
        sign_l=sign_l, sign_r=sign_r,
        W_scale=bs_dict["W_scale"],
        shape=W.shape,
        meta={"label": label},
    )

    file_size = os.path.getsize(out_path)
    bits_per_weight = file_size * 8 / W.size
    print(f"  saved to {out_path}, {file_size/1024:.0f} KB ({bits_per_weight:.2f} bits/weight)")

    # Load + dequant
    saved = load_quantized(out_path)
    Wh_dequant = dequant_target(saved, decode)

    # Compare
    direct_64 = Wh_direct.astype(np.float64)
    dequant_64 = Wh_dequant.astype(np.float64)
    max_diff = float(np.abs(direct_64 - dequant_64).max())
    bit_exact = np.array_equal(direct_64, dequant_64)
    rel_err = float(np.linalg.norm(direct_64 - dequant_64) / np.linalg.norm(direct_64))

    print(f"  Wh_direct  range: [{Wh_direct.min():+.4f}, {Wh_direct.max():+.4f}]")
    print(f"  Wh_dequant range: [{Wh_dequant.min():+.4f}, {Wh_dequant.max():+.4f}]")
    print(f"  max abs diff: {max_diff:.2e}")
    print(f"  bit-exact: {bit_exact}")
    print(f"  relative diff: {rel_err:.2e}")

    ok = max_diff < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] round-trip exact (max diff < 1e-12)")
    return ok, bits_per_weight


def test_synthetic():
    print("\nA.8.1.1: synthetic 128x128 round-trip")
    print("-" * 60)
    rng = np.random.default_rng(0)
    m, n = 128, 128
    W = rng.standard_normal((m, n)).astype(np.float32) * 0.02
    A = rng.standard_normal((n, n)).astype(np.float32)
    H = (A @ A.T / n + 0.1 * np.eye(n)).astype(np.float32)
    sign_l = make_sign_vector(m, seed=1)
    sign_r = make_sign_vector(n, seed=2)
    ok, bpw = _round_trip(W, H, sign_l, sign_r, "synthetic_128")
    return ok


def test_real_expert():
    print("\nA.8.1.2: real L8 E0 down_proj round-trip")
    print("-" * 60)

    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    key = "model.layers.8.mlp.experts.0.down_proj.weight"
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        W = f.get_tensor(key).astype(np.float32)

    H = torch.load(
        os.path.join(HESSIAN_DIR, "L08", "expert_00_down.pt"),
        weights_only=True,
    )["H"].numpy()

    sign_l = make_sign_vector(W.shape[0], seed=8001)
    sign_r = make_sign_vector(W.shape[1], seed=8002)
    ok, bpw = _round_trip(W, H, sign_l, sign_r, "L8_E0_down")
    return ok, bpw


def main():
    print("=" * 60)
    print("Tripwire A.8.1: quantized format save/load")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    results = []
    results.append(("synthetic round-trip", test_synthetic()))
    real_ok, real_bpw = test_real_expert()
    results.append(("real L8 E0 round-trip", real_ok))

    print("\n" + "=" * 60)
    print(f"  Real expert disk size: {real_bpw:.2f} bits/weight")
    print(f"  Target: ~2.0 bits/weight (entropy) ~4.0 bits/weight (uint8 storage)")
    storage_ok = real_bpw < 5.0
    results.append(("storage size sane", storage_ok))

    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.8.1 GATE: PASS — quantized format verified.")
        print("Ready for A.8.2 (single-layer driver).")
        sys.exit(0)
    else:
        print("A.8.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()