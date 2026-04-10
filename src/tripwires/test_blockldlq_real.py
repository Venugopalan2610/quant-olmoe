"""Tripwire A.7.3.C.3: BlockLDLQ on real OLMoE Hessians via CUDA Viterbi.

A.7.3.C.3.1: 128x128 sanity (CUDA path matches numpy path on synthetic)
A.7.3.C.3.2: L8 E0 gate_proj (1024x2048, easy Hessian, cond ~10^3)
A.7.3.C.3.3: L0 E0 gate_proj (1024x2048, hard Hessian, cond ~10^10)
A.7.3.C.3.4: L8 E0 down_proj  (2048x1024, easy Hessian)

Each test reports: wall clock, proxy loss, per-tile MSE, W reconstruction
relative error.

Run: python -m src.tripwires.test_blockldlq_real
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

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HESSIAN_DIR = "cache/hessians"
LUT_PATH = "cache/codes/hyb_lut_init.npy"


def _decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def _load_expert_weight(layer, expert, proj):
    """Load one expert projection from safetensors. proj in {gate_proj, up_proj, down_proj}."""
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        return f.get_tensor(key).astype(np.float32)


def _load_expert_hessian(layer, expert, kind):
    """Load expert Hessian. kind in {gate_up, down}."""
    path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"expert_{expert:02d}_{kind}.pt")
    return torch.load(path, weights_only=True)["H"].numpy()


def _run_real(layer, expert, proj, label):
    """Quantize one real OLMoE expert weight using its Hessian."""
    print(f"\n  --- {label} ---")
    W = _load_expert_weight(layer, expert, proj)
    print(f"  W shape: {W.shape}, dtype: {W.dtype}")
    print(f"  W stats: mean={W.mean():+.5f} std={W.std():.5f} max|.|={np.abs(W).max():.4f}")

    h_kind = "down" if proj == "down_proj" else "gate_up"
    H = _load_expert_hessian(layer, expert, h_kind)
    print(f"  H shape: {H.shape}")
    eigs = np.linalg.eigvalsh(H[:512, :512])
    cond_top = float(eigs.max() / max(eigs.min(), 1e-30))
    print(f"  H[:512,:512] cond: {cond_top:.2e} (full cond likely larger)")

    sign_l = make_sign_vector(W.shape[0], seed=layer * 1000 + expert)
    sign_r = make_sign_vector(W.shape[1], seed=layer * 1000 + expert + 1)
    decode = _decoder()

    t0 = time.time()
    Wh, Wh_tilde, proxy, diag = blockldlq(
        W, H, sign_l, sign_r, decode,
        L_bits=16, k=2, V=2, Tx=16, Ty=16, damp=0.01,
        use_cuda=True,
        return_diagnostics=True,
    )
    dt = time.time() - t0

    rel_err = float(np.linalg.norm(Wh - W) / np.linalg.norm(W))
    max_recon = float(np.abs(Wh).max())
    max_orig = float(np.abs(W).max())

    print(f"  wall clock: {dt:.1f}s")
    print(f"  W_scale: {diag['W_scale']:.5f}")
    print(f"  per-tile MSE range (unit basis): "
          f"[{min(diag['per_block_mse_in_tile']):.4f}, "
          f"{max(diag['per_block_mse_in_tile']):.4f}]")
    print(f"  proxy loss: {proxy:.4e}")
    print(f"  ||Wh - W|| / ||W||: {rel_err:.4f}")
    print(f"  max |Wh|: {max_recon:.4f}  max |W|: {max_orig:.4f}")

    finite_ok = bool(np.isfinite(Wh).all())
    scale_ok = max_recon < 5 * max_orig
    rel_ok = rel_err < 1.5  # 2-bit quant on weights ~ rel err around 1.0
    proxy_ok = 0 < proxy < 10.0  # generous; absolute scale depends on H

    print(f"  [{'PASS' if finite_ok else 'FAIL'}] reconstruction finite")
    print(f"  [{'PASS' if scale_ok else 'FAIL'}] scale preserved")
    print(f"  [{'PASS' if rel_ok else 'FAIL'}] relative err < 1.5")
    print(f"  [{'PASS' if proxy_ok else 'FAIL'}] proxy loss in (0, 10)")

    return finite_ok and scale_ok and rel_ok and proxy_ok


def test_synthetic_cuda_vs_numpy():
    """Sanity check: CUDA path produces same result as numpy path on tiny problem."""
    print("\nA.7.3.C.3.1: CUDA path == numpy path on 128x128 synthetic")
    print("-" * 60)

    rng = np.random.default_rng(0)
    m, n = 128, 128
    W = rng.standard_normal((m, n)).astype(np.float64) * 0.02
    A_rand = rng.standard_normal((n, n))
    H = A_rand @ A_rand.T / n + 0.1 * np.eye(n)

    sign_l = make_sign_vector(m, seed=1)
    sign_r = make_sign_vector(n, seed=2)
    decode = _decoder()

    Wh_np, _, proxy_np = blockldlq(W, H, sign_l, sign_r, decode, use_cuda=False)
    Wh_cu, _, proxy_cu = blockldlq(W, H, sign_l, sign_r, decode, use_cuda=True)

    max_diff = float(np.abs(Wh_np - Wh_cu).max())
    print(f"  numpy proxy: {proxy_np:.4e}")
    print(f"  cuda  proxy: {proxy_cu:.4e}")
    print(f"  max abs diff Wh_np vs Wh_cu: {max_diff:.2e}")

    ok = max_diff < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] CUDA path matches numpy path")
    return ok


def main():
    print("=" * 60)
    print("Tripwire A.7.3.C.3: BlockLDLQ on real OLMoE experts")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    results = []
    results.append(("synthetic CUDA == numpy", test_synthetic_cuda_vs_numpy()))
    results.append(("L8 E0 gate_proj (easy)",
                    _run_real(8, 0, "gate_proj", "L8 E0 gate_proj")))
    results.append(("L0 E0 gate_proj (hard)",
                    _run_real(0, 0, "gate_proj", "L0 E0 gate_proj")))
    results.append(("L8 E0 down_proj (easy)",
                    _run_real(8, 0, "down_proj", "L8 E0 down_proj")))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.3.C.3 GATE: PASS — BlockLDLQ on real data verified.")
        print("Ready for A.8 (full-model quantization).")
        sys.exit(0)
    else:
        print("A.7.3.C.3 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()