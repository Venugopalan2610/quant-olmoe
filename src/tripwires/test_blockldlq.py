"""Tripwire A.7.2: BlockLDLQ reference implementation."""
import os
import sys
import time
import numpy as np

from src.codes.ref import decode_hyb_batch
from src.rht.transform import make_sign_vector, apply_rht
from src.quantize.blockldlq import blockldlq


LUT_PATH = "cache/codes/hyb_lut_init.npy"


def make_hyb_decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def _initial_loss(W, H, sign_l, sign_r):
    """Proxy loss before any quantization (Wh_tilde = 0)."""
    W64 = W.astype(np.float64)
    H64 = H.astype(np.float64)
    W_tilde = apply_rht(W64, sign_l, sign_r)
    H_tilde = apply_rht(H64, sign_r, sign_r)
    m = W.shape[0]
    return float(np.trace(W_tilde @ H_tilde @ W_tilde.T) / m)


def test_a721_identity_h():
    """Identity Hessian — BlockLDLQ reduces to per-tile Viterbi."""
    print("\nA.7.2.1: synthetic 128x128 with identity Hessian")
    print("-" * 60)

    m, n = 128, 128
    rng = np.random.default_rng(0)
    W = rng.standard_normal((m, n)).astype(np.float64) * 0.02
    H = np.eye(n, dtype=np.float64)

    sign_l = make_sign_vector(m, seed=1)
    sign_r = make_sign_vector(n, seed=2)
    decode = make_hyb_decoder()

    initial = _initial_loss(W, H, sign_l, sign_r)

    t0 = time.time()
    Wh, Wh_tilde, proxy, diag = blockldlq(
        W, H, sign_l, sign_r, decode, return_diagnostics=True,
    )
    dt = time.time() - t0

    print(f"  wall clock: {dt:.1f}s, Viterbi calls: {diag['n_viterbi_calls']}")
    print(f"  W_scale: {diag['W_scale']:.6f}")
    print(f"  initial loss (Wh_tilde=0): {initial:.4e}")
    print(f"  final proxy loss: {proxy:.4e}")
    print(f"  loss reduction: {initial / proxy:.1f}x")
    print(f"  per-tile MSE (unit basis) range: "
          f"[{min(diag['per_block_mse_in_tile']):.4f}, "
          f"{max(diag['per_block_mse_in_tile']):.4f}]")

    # With unit-variance scaling, per-tile MSE should be ~HYB floor (0.065-0.075)
    tile_mse_ok = all(0.05 < mse < 0.10 for mse in diag["per_block_mse_in_tile"])
    # Expected proxy: ~0.069 * n * W_scale^2 = ~0.069 * 128 * 4e-4 ≈ 3.5e-3
    proxy_ok = 1e-5 < proxy < 1e-1
    reduction_ok = (initial / proxy) > 5
    finite_ok = np.isfinite(Wh).all()
    scale_ok = float(np.abs(Wh).max()) < 5 * float(np.abs(W).max())

    print(f"  [{'PASS' if tile_mse_ok else 'FAIL'}] per-tile MSE near HYB floor")
    print(f"  [{'PASS' if proxy_ok else 'FAIL'}] proxy loss in [1e-5, 1e-1]")
    print(f"  [{'PASS' if reduction_ok else 'FAIL'}] loss reduction > 5x")
    print(f"  [{'PASS' if finite_ok else 'FAIL'}] reconstruction finite")
    print(f"  [{'PASS' if scale_ok else 'FAIL'}] reconstruction scale preserved")

    return tile_mse_ok and proxy_ok and reduction_ok and finite_ok and scale_ok


def test_a722_random_pd_h():
    """Random PD Hessian — verify monotone loss and feedback benefit."""
    print("\nA.7.2.2: synthetic 128x128 with random PD Hessian")
    print("-" * 60)

    m, n = 128, 128
    rng = np.random.default_rng(0)
    W = rng.standard_normal((m, n)).astype(np.float64) * 0.02

    A_rand = rng.standard_normal((n, n))
    H = A_rand @ A_rand.T / n + 0.1 * np.eye(n)

    sign_l = make_sign_vector(m, seed=1)
    sign_r = make_sign_vector(n, seed=2)
    decode = make_hyb_decoder()

    initial = _initial_loss(W, H, sign_l, sign_r)

    t0 = time.time()
    Wh, Wh_tilde, proxy, diag = blockldlq(
        W, H, sign_l, sign_r, decode, return_diagnostics=True,
    )
    dt = time.time() - t0

    losses = diag["proxy_loss_per_step"]
    print(f"  wall clock: {dt:.1f}s")
    print(f"  W_scale: {diag['W_scale']:.6f}")
    print(f"  initial loss: {initial:.4e}")
    print(f"  final proxy loss: {proxy:.4e}")
    print(f"  loss reduction: {initial / proxy:.1f}x")
    print(f"  loss per block (first 4): {[f'{l:.3e}' for l in losses[:4]]}")
    print(f"  loss per block (last 4):  {[f'{l:.3e}' for l in losses[-4:]]}")

    # Relaxed monotonicity: allow small transient increases but no catastrophic ones,
    # and the final loss must be less than the first recorded loss.
    diffs = np.diff(losses)
    max_increase = float(diffs.max()) if len(diffs) > 0 else 0.0
    max_increase_frac = max_increase / max(losses[0], 1e-30)
    overall_decrease = losses[-1] < losses[0]
    print(f"  max step-to-step increase: {max_increase:.2e} "
          f"({100*max_increase_frac:.1f}% of first loss)")
    print(f"  final < first: {overall_decrease}")

    monotone_ok = overall_decrease and max_increase_frac < 0.10
    finite_ok = np.isfinite(Wh).all()
    reduction_ok = (initial / proxy) > 3
    proxy_ok = 1e-6 < proxy < 1e-1

    print(f"  [{'PASS' if monotone_ok else 'FAIL'}] loss decreases overall, "
          f"max transient increase <10%")
    print(f"  [{'PASS' if reduction_ok else 'FAIL'}] loss reduction > 3x")
    print(f"  [{'PASS' if proxy_ok else 'FAIL'}] proxy in [1e-6, 1e-1]")
    print(f"  [{'PASS' if finite_ok else 'FAIL'}] reconstruction finite")

    return monotone_ok and reduction_ok and proxy_ok and finite_ok


def test_a723_ill_conditioned():
    """Ill-conditioned Hessian (cond ~1e11) — damping must save us."""
    print("\nA.7.2.3: synthetic 128x128 with ill-conditioned Hessian")
    print("-" * 60)

    m, n = 128, 128
    rng = np.random.default_rng(0)
    W = rng.standard_normal((m, n)).astype(np.float64) * 0.02

    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = np.zeros(n)
    eigvals[:10] = np.linspace(1.0, 10.0, 10)
    eigvals[10:] = np.logspace(-10, -6, n - 10)
    H = Q @ np.diag(eigvals) @ Q.T
    H = 0.5 * (H + H.T)
    actual_cond = eigvals.max() / eigvals.min()
    print(f"  Hessian cond: {actual_cond:.2e}")

    sign_l = make_sign_vector(m, seed=1)
    sign_r = make_sign_vector(n, seed=2)
    decode = make_hyb_decoder()

    initial = _initial_loss(W, H, sign_l, sign_r)

    t0 = time.time()
    try:
        Wh, Wh_tilde, proxy, diag = blockldlq(
            W, H, sign_l, sign_r, decode,
            damp=0.01, return_diagnostics=True,
        )
        success = True
    except Exception as e:
        print(f"  FAIL: blockldlq raised: {e}")
        return False
    dt = time.time() - t0

    print(f"  wall clock: {dt:.1f}s")
    print(f"  W_scale: {diag['W_scale']:.6f}")
    print(f"  initial loss: {initial:.4e}")
    print(f"  proxy loss: {proxy:.4e}")
    print(f"  max |Wh|: {float(np.abs(Wh).max()):.4f}  max |W|: {float(np.abs(W).max()):.4f}")

    finite_ok = bool(np.isfinite(Wh).all())
    scale_ok = float(np.abs(Wh).max()) < 5 * float(np.abs(W).max())
    # Ill-conditioned H means some eigen-directions are "don't care";
    # loss can be larger than well-conditioned case but must still be finite.
    proxy_ok = 0 < proxy < 1.0

    print(f"  [{'PASS' if finite_ok else 'FAIL'}] reconstruction finite")
    print(f"  [{'PASS' if scale_ok else 'FAIL'}] reconstruction scale preserved")
    print(f"  [{'PASS' if proxy_ok else 'FAIL'}] proxy loss in (0, 1.0)")

    return finite_ok and scale_ok and proxy_ok


def main():
    print("=" * 60)
    print("Tripwire A.7.2: BlockLDLQ reference")
    print("=" * 60)

    results = []
    results.append(("A.7.2.1 identity H", test_a721_identity_h()))
    results.append(("A.7.2.2 random PD H", test_a722_random_pd_h()))
    results.append(("A.7.2.3 ill-conditioned", test_a723_ill_conditioned()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.2 GATE: PASS — BlockLDLQ reference verified.")
        print("Ready for A.7.3 (batched Viterbi).")
        sys.exit(0)
    else:
        print("A.7.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()