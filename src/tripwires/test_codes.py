"""Tripwire A.1: decode function correctness.

A.1.1: Marginal distribution (mean, var, kurtosis) within Gaussian bounds
A.1.2: Decorrelation (visual scatter plot, no diagonal stripes)
A.1.3: Bitshift neighborhood spread (4 successors span the distribution)
A.1.4: HYB LUT initialization quality

Run: python -m src.tripwires.test_codes
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.codes.ref import decode_1mad_batch, decode_3inst_batch, decode_hyb_batch
from src.codes.lut_init import init_hyb_lut, lut_mse


PLOTS_DIR = "./plots/codes"
os.makedirs(PLOTS_DIR, exist_ok=True)


def check_marginal(name, samples):
    """A.1.1 — verify samples look like N(0,1)."""
    mean = float(np.mean(samples))
    var = float(np.var(samples))
    kurt = float(stats.kurtosis(samples, fisher=False))  # 3 = Gaussian

    mean_ok = abs(mean) < 0.05
    var_ok = 0.85 < var < 1.15
    kurt_ok = 2.6 < kurt < 3.4

    status = "PASS" if (mean_ok and var_ok and kurt_ok) else "FAIL"
    print(f"  [{status}] {name}: mean={mean:+.4f} var={var:.4f} kurtosis={kurt:.3f}")
    return mean_ok and var_ok and kurt_ok


def check_decorrelation(name, samples_a, samples_b):
    """A.1.2 — verify (s_t, s_{t+1}) pairs look like independent Gaussians.

    Save scatter plot, compute Pearson correlation. |r| < 0.05 = pass.
    """
    r = float(np.corrcoef(samples_a, samples_b)[0, 1])
    ok = abs(r) < 0.05

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(samples_a[:5000], samples_b[:5000], s=0.5, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: bitshift-adjacent states\nPearson r = {r:+.4f}")
    ax.set_xlabel("decode(state)")
    ax.set_ylabel("decode(shift(state))")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_scatter.png"), dpi=100)
    plt.close()

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: |r|={abs(r):.4f} (saved scatter to ./plots/codes/)")
    return ok


def check_neighborhood_spread(name, decode_batch_fn, L=16, k=2, n_states=10_000, seed=0):
    """A.1.3 — for random L-bit states, the 4 successors via bitshift should
    span a meaningful fraction of N(0,1).

    Bitshift rule: next = ((s << k) | c) & ((1<<L) - 1) for c in {0..2^k - 1}.
    Compute spread (max - min) of decoded successors per state, average.
    """
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 1 << L, size=n_states, dtype=np.uint32)

    spreads = []
    mask = np.uint32((1 << L) - 1)
    for c in range(1 << k):
        succ = ((states << np.uint32(k)) | np.uint32(c)) & mask
        decoded = decode_batch_fn(succ)
        # decoded is shape (n_states,) for V=1 codes
        if c == 0:
            stack = decoded[:, None]
        else:
            stack = np.concatenate([stack, decoded[:, None]], axis=1)

    spread = stack.max(axis=1) - stack.min(axis=1)
    mean_spread = float(spread.mean())
    median_spread = float(np.median(spread))

    # Expected: a Gaussian's typical 4-sample spread is ~1.5σ, so we want
    # mean spread > 1.0 to be safe. If it's < 0.5, the code is correlated.
    ok = mean_spread > 1.0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: mean spread of 4 successors = {mean_spread:.3f} "
          f"(median {median_spread:.3f})")
    return ok


def main():
    print("=" * 60)
    print("Tripwire A.1: decode function correctness")
    print("=" * 60)

    n_samples = 1_000_000
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # A.1.4 first — initialize LUT (needed for HYB tests below)
    # ------------------------------------------------------------------
    print("\nA.1.4: HYB LUT initialization")
    print("-" * 60)
    lut = init_hyb_lut(Q=9, n_samples=1_000_000, seed=0)
    lut_distortion = lut_mse(lut)
    # 9-bit 2D Gaussian quantization Lloyd-Max bound is around 0.025-0.035
    lut_ok = 0.004 < lut_distortion < 0.05
    status = "PASS" if lut_ok else "FAIL"
    print(f"  [{status}] LUT shape={lut.shape}, MSE={lut_distortion:.4f} "
          f"(expected 0.004–0.05)")

    # Save LUT plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(lut[:, 0], lut[:, 1], s=4)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(f"HYB LUT (Q=9, 512 centroids)\nMSE on 2D Gaussian = {lut_distortion:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hyb_lut_init.png"), dpi=100)
    plt.close()
    print("  Saved LUT plot to ./plots/codes/hyb_lut_init.png")

    # Save LUT for use in A.2 onwards
    os.makedirs("cache/codes", exist_ok=True)
    np.save("cache/codes/hyb_lut_init.npy", lut)
    print("  Saved LUT to cache/codes/hyb_lut_init.npy")

    # ------------------------------------------------------------------
    # A.1.1 — marginal distributions
    # ------------------------------------------------------------------
    print("\nA.1.1: Marginal distributions (1M samples each)")
    print("-" * 60)
    states = rng.integers(0, 2 ** 16, size=n_samples, dtype=np.uint32)

    s_1mad = decode_1mad_batch(states)
    m1 = check_marginal("1MAD", s_1mad)

    s_3inst = decode_3inst_batch(states)
    m2 = check_marginal("3INST", s_3inst)

    hyb_pairs = decode_hyb_batch(states, lut, Q=9)
    m3a = check_marginal("HYB (component 0)", hyb_pairs[:, 0])
    m3b = check_marginal("HYB (component 1)", hyb_pairs[:, 1])

    # ------------------------------------------------------------------
    # A.1.2 — decorrelation: BITSHIFT-adjacent state pairs
    # The trellis never sees +1-adjacent states; it sees shift-adjacent
    # ones via next = ((s << kV) | c) & mask. Test that relation.
    # ------------------------------------------------------------------
    print("\nA.1.2: Decorrelation (bitshift-adjacent states, k=2, V=1, L=16)")
    print("-" * 60)
    n_pairs = 50_000
    states_a = rng.integers(0, 2 ** 16, size=n_pairs, dtype=np.uint32)
    new_bits = rng.integers(0, 4, size=n_pairs, dtype=np.uint32)  # k=2, so 0..3
    states_b = ((states_a << np.uint32(2)) | new_bits) & np.uint32(0xFFFF)

    s1_a = decode_1mad_batch(states_a)
    s1_b = decode_1mad_batch(states_b)
    d1 = check_decorrelation("1mad", s1_a, s1_b)

    s2_a = decode_3inst_batch(states_a)
    s2_b = decode_3inst_batch(states_b)
    d2 = check_decorrelation("3inst", s2_a, s2_b)

    h_a = decode_hyb_batch(states_a, lut, Q=9)[:, 0]
    h_b = decode_hyb_batch(states_b, lut, Q=9)[:, 0]
    d3 = check_decorrelation("hyb", h_a, h_b)

    # ------------------------------------------------------------------
    # A.1.3 — bitshift successor spread
    # ------------------------------------------------------------------
    print("\nA.1.3: Bitshift successor spread (k=2, 4 successors per state)")
    print("-" * 60)
    n1 = check_neighborhood_spread("1MAD", decode_1mad_batch, L=16, k=2)
    n2 = check_neighborhood_spread("3INST", decode_3inst_batch, L=16, k=2)
    n3 = check_neighborhood_spread(
        "HYB(comp0)",
        lambda s: decode_hyb_batch(s, lut, Q=9)[:, 0],
        L=16,
        k=2,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    all_ok = all([m1, m2, m3a, m3b, d1, d2, d3, n1, n2, n3, lut_ok])
    if all_ok:
        print("A.1 GATE: PASS — all decoders verified, LUT initialized.")
        print("Ready for A.2 (Viterbi).")
        sys.exit(0)
    else:
        print("A.1 GATE: FAIL — fix failing tripwires before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()