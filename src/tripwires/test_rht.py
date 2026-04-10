"""Tripwire A.3: RHT correctness and incoherence properties.

A.3.1: Roundtrip — apply_rht then apply_inverse_rht should recover input
A.3.2: Gaussianification — RHT of a real OLMoE weight matrix should be
       approximately Gaussian (kurtosis ~3)
A.3.3: Incoherence bound — max abs entry of W̃ bounded by paper's formula

Run: python -m src.tripwires.test_rht
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch

from src.rht.transform import apply_rht, apply_inverse_rht, make_sign_vector


PLOTS_DIR = "plots/rht"
os.makedirs(PLOTS_DIR, exist_ok=True)


def test_a31_roundtrip():
    """A.3.1 — RHT then inverse RHT should recover the input."""
    print("\nA.3.1: RHT roundtrip")
    print("-" * 60)

    rng = np.random.default_rng(0)
    m, n = 1024, 2048
    W = rng.standard_normal((m, n)).astype(np.float32)

    sign_left = make_sign_vector(m, seed=1)
    sign_right = make_sign_vector(n, seed=2)

    W_tilde = apply_rht(W, sign_left, sign_right)
    W_recon = apply_inverse_rht(W_tilde, sign_left, sign_right)

    max_abs_diff = float(np.max(np.abs(W - W_recon)))
    rel_err = max_abs_diff / float(np.max(np.abs(W)))

    print(f"  Shape: {W.shape}, dtype: {W.dtype}")
    print(f"  Max abs diff: {max_abs_diff:.2e}")
    print(f"  Relative err: {rel_err:.2e}")

    ok = max_abs_diff < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] roundtrip < 1e-4")
    return ok


def test_a32_gaussianification():
    """A.3.2 — RHT of a real OLMoE expert weight matrix should look Gaussian.

    Loads one tensor directly from the safetensors shard to avoid the 7-minute
    full-model load.
    """
    print("\nA.3.2: Gaussianification of OLMoE weight matrix")
    print("-" * 60)

    import json
    from safetensors import safe_open

    model_dir = "cache/model/olmoe-1b-7b-0125"
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)

    # Find any gate_proj tensor in layer 0. The exact key depends on whether
    # HF stores experts as fused or per-expert.
    candidates = [
        k for k in index["weight_map"]
        if "layers.0." in k and "experts" in k and "gate_proj" in k
    ]
    if not candidates:
        # Fused storage: look for the stacked expert weights
        candidates = [
            k for k in index["weight_map"]
            if "layers.0.mlp.experts" in k and "gate" in k
        ]
    assert candidates, f"No expert gate weight found in layer 0. Available keys (sample): {list(index['weight_map'].keys())[:5]}"

    key = sorted(candidates)[0]
    shard = index["weight_map"][key]
    print(f"  Loading {key} from {shard}")

    with safe_open(os.path.join(model_dir, shard), framework="numpy") as f:
        W_full = f.get_tensor(key)

    # Handle both possible storage formats:
    #   per-expert: shape (intermediate, hidden) e.g. (1024, 2048)
    #   fused:      shape (num_experts, intermediate, hidden) e.g. (64, 1024, 2048)
    if W_full.ndim == 3:
        W = W_full[0].astype(np.float32)
        print(f"  Fused storage detected, sliced expert 0: shape={W.shape}")
    else:
        W = W_full.astype(np.float32)
        print(f"  Per-expert storage: shape={W.shape}")

    kurt_before = float(stats.kurtosis(W.flatten(), fisher=False))
    std_before = float(np.std(W))

    sign_left = make_sign_vector(W.shape[0], seed=1)
    sign_right = make_sign_vector(W.shape[1], seed=2)
    W_tilde = apply_rht(W, sign_left, sign_right)

    kurt_after = float(stats.kurtosis(W_tilde.flatten(), fisher=False))
    std_after = float(np.std(W_tilde))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(W.flatten(), bins=200, density=True, alpha=0.7)
    axes[0].set_title(f"Before RHT\nkurtosis={kurt_before:.2f}, std={std_before:.4f}")
    axes[0].set_xlabel("weight value")

    x_range = np.linspace(W_tilde.min(), W_tilde.max(), 200)
    gauss = (1 / (std_after * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std_after) ** 2)
    axes[1].hist(W_tilde.flatten(), bins=200, density=True, alpha=0.7, label="W̃")
    axes[1].plot(x_range, gauss, "r-", lw=1, label="N(0, σ²)")
    axes[1].set_title(f"After RHT\nkurtosis={kurt_after:.2f}, std={std_after:.4f}")
    axes[1].set_xlabel("weight value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rht_gaussianification.png"), dpi=100)
    plt.close()
    print(f"  Saved histogram to {PLOTS_DIR}/rht_gaussianification.png")

    print(f"  Kurtosis before: {kurt_before:.3f}")
    print(f"  Kurtosis after:  {kurt_after:.3f} (target: ~3.0)")
    print(f"  Std before:      {std_before:.6f}")
    print(f"  Std after:       {std_after:.6f}")

    std_ok = abs(std_after - std_before) / std_before < 0.05
    kurt_ok = 2.5 < kurt_after < 4.0

    print(f"  [{'PASS' if std_ok else 'FAIL'}] std preserved (RHT is orthogonal)")
    print(f"  [{'PASS' if kurt_ok else 'FAIL'}] kurtosis close to Gaussian")
    return std_ok and kurt_ok


def test_a33_incoherence_bound():
    """A.3.3 — RHT-processed weights satisfy the µ-incoherence bound.

    Paper definition: max_ij |W̃_ij| <= µ * ||W||_F / sqrt(mn)
    where µ = 2 * sqrt(log(4mn/δ)) with high probability.

    For δ = 1/100 and a 1024x2048 matrix, µ ≈ 2*sqrt(log(819200)) ≈ 7.4.
    The actual max should be a few sigma below this bound.
    """
    print("\nA.3.3: µ-incoherence bound")
    print("-" * 60)

    rng = np.random.default_rng(0)
    m, n = 1024, 2048
    # Use a weight with outliers to make this nontrivial
    W = rng.standard_normal((m, n)).astype(np.float32)
    W[0, 0] = 50.0  # synthetic outlier
    W[100, 200] = -30.0  # another

    frob = float(np.linalg.norm(W))
    max_before = float(np.max(np.abs(W)))
    bound_term = frob / np.sqrt(m * n)
    mu_before = max_before / bound_term

    sign_left = make_sign_vector(m, seed=1)
    sign_right = make_sign_vector(n, seed=2)
    W_tilde = apply_rht(W, sign_left, sign_right)

    max_after = float(np.max(np.abs(W_tilde)))
    mu_after = max_after / bound_term  # bound_term unchanged since RHT preserves Frobenius

    # Theoretical bound: 2*sqrt(log(4mn/0.01))
    delta = 0.01
    mu_bound = 2 * np.sqrt(np.log(4 * m * n / delta))

    print(f"  Shape: ({m}, {n}), Frobenius norm: {frob:.4f}")
    print(f"  Bound term ||W||_F/sqrt(mn): {bound_term:.6f}")
    print(f"  Before RHT: max |W| = {max_before:.4f}, µ = {mu_before:.2f}")
    print(f"  After RHT:  max |W̃| = {max_after:.4f}, µ = {mu_after:.2f}")
    print(f"  Theoretical bound (δ=1/100): µ ≤ {mu_bound:.2f}")

    # The big check: outlier-driven µ before RHT should be huge; after RHT
    # it should be within the theoretical bound
    # Demand at least 10x improvement in incoherence from RHT
    huge_before = mu_before > 10 * mu_after
    bounded_after = mu_after < mu_bound

    print(f"  [{'PASS' if huge_before else 'FAIL'}] outliers make µ_before huge")
    print(f"  [{'PASS' if bounded_after else 'FAIL'}] µ_after within theoretical bound")
    return huge_before and bounded_after


def main():
    print("=" * 60)
    print("Tripwire A.3: RHT correctness and incoherence")
    print("=" * 60)

    results = []
    results.append(("A.3.1 roundtrip", test_a31_roundtrip()))
    results.append(("A.3.2 gaussianification", test_a32_gaussianification()))
    results.append(("A.3.3 incoherence bound", test_a33_incoherence_bound()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.3 GATE: PASS — RHT verified.")
        print("Ready for A.4 (OLMoE adapter).")
        sys.exit(0)
    else:
        print("A.3 GATE: FAIL — fix failing tripwires before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()