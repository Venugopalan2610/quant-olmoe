"""Tripwire A.7.0: V-generic Viterbi for V=2 HYB.

A.7.0.1: V=1 encoder matches the V-specific one on a random sequence
         (backward compat check)
A.7.0.2: V=2 HYB hits paper Table 1/11 MSE on synthetic Gaussians
A.7.0.3: V=2 encode/decode round-trip bit-exact
A.7.0.4: V=2 tail-biting close to non-tail-biting MSE

Run: python -m src.tripwires.test_viterbi_v2
"""
import sys
import numpy as np

from src.codes.ref import decode_1mad_batch, decode_hyb_batch
from src.viterbi.encode import (
    viterbi_encode,
    viterbi_encode_v,
    viterbi_decode_v,
    viterbi_encode_tailbiting_v,
)


def test_a700_backcompat():
    """A.7.0.1 — V-generic with V=1 matches the V=1-specific encoder."""
    print("\nA.7.0.1: V-generic V=1 backward compat with V=1-specific")
    print("-" * 60)
    rng = np.random.default_rng(0)
    seq = rng.standard_normal(256).astype(np.float32)

    _, _, recon_v1, mse_v1 = viterbi_encode(seq, L=16, k=2, decode_fn=decode_1mad_batch)
    _, _, recon_vg, mse_vg = viterbi_encode_v(seq, L=16, k=2, V=1, decode_fn=decode_1mad_batch)

    match = np.allclose(recon_v1, recon_vg)
    print(f"  V=1-specific MSE: {mse_v1:.6f}")
    print(f"  V-generic V=1 MSE: {mse_vg:.6f}")
    print(f"  Reconstructions match: {match}")
    print(f"  [{'PASS' if match else 'FAIL'}] backward compat")
    return match


def test_a701_hyb_v2_table1():
    """A.7.0.2 — V=2 HYB reproduces Table 1 MSE on iid Gaussian."""
    print("\nA.7.0.2: V=2 HYB Table 1 reproduction")
    print("-" * 60)

    lut = np.load("cache/codes/hyb_lut_init.npy")  # (512, 2)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)  # (N, 2)

    rng = np.random.default_rng(42)
    n_seqs = 32
    T = 256
    mses = []
    for _ in range(n_seqs):
        seq = rng.standard_normal(T).astype(np.float32)
        _, _, _, mse = viterbi_encode_v(seq, L=16, k=2, V=2, decode_fn=hyb_v2)
        mses.append(mse)

    mean_mse = float(np.mean(mses))
    std_mse = float(np.std(mses))

    # Paper Table 1 HYB MSE = 0.069, DR = 0.063
    # V=2 should be slightly better than V=1 HYB (0.0675) per paper Table 11
    ok = 0.062 <= mean_mse <= 0.075
    print(f"  V=2 HYB MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"  Paper Table 1: 0.069")
    print(f"  Rate-distortion bound: 0.063")
    print(f"  [{'PASS' if ok else 'FAIL'}] in valid range [0.062, 0.075]")
    return ok


def test_a702_v2_roundtrip():
    """A.7.0.3 — V=2 encode/decode round-trip bit-exact."""
    print("\nA.7.0.3: V=2 round-trip")
    print("-" * 60)

    lut = np.load("cache/codes/hyb_lut_init.npy")
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)

    rng = np.random.default_rng(1)
    seq = rng.standard_normal(256).astype(np.float32)

    bitstream, start_state, recon, mse = viterbi_encode_v(
        seq, L=16, k=2, V=2, decode_fn=hyb_v2
    )
    replayed = viterbi_decode_v(bitstream, start_state, L=16, k=2, V=2, decode_fn=hyb_v2)

    max_diff = float(np.max(np.abs(recon - replayed)))
    ok = max_diff == 0.0
    print(f"  MSE: {mse:.4f}")
    print(f"  recon vs replayed max abs diff: {max_diff:.2e}")
    print(f"  bitstream shape: {bitstream.shape} (expected (128,))")
    print(f"  [{'PASS' if ok else 'FAIL'}] bit-exact")
    return ok


def test_a703_v2_tailbiting():
    """A.7.0.4 — V=2 tail-biting close to non-tail-biting."""
    print("\nA.7.0.4: V=2 tail-biting")
    print("-" * 60)

    lut = np.load("cache/codes/hyb_lut_init.npy")
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)

    rng = np.random.default_rng(42)
    n_seqs = 16
    T = 256

    base_mses = []
    tb_mses = []
    overlap_ok = 0
    for _ in range(n_seqs):
        seq = rng.standard_normal(T).astype(np.float32)
        _, _, _, mse_base = viterbi_encode_v(seq, L=16, k=2, V=2, decode_fn=hyb_v2)
        bs_tb, start_tb, _recon_tb, mse_tb, overlap = viterbi_encode_tailbiting_v(
            seq, L=16, k=2, V=2, decode_fn=hyb_v2
        )
        base_mses.append(mse_base)
        tb_mses.append(mse_tb)

        # Verify tail-biting property: s_0.top(L-kV) == s_{last}.bottom(L-kV) == overlap
        overlap_bits = 16 - 2 * 2  # L - kV = 12
        mask = (1 << 16) - 1
        s = start_tb
        s0 = None
        s_last = None
        for idx, c in enumerate(bs_tb):
            s = ((s << 4) | int(c)) & mask
            if idx == 0:
                s0 = s
            s_last = s
        s0_top = s0 >> 4
        s_last_bot = s_last & ((1 << overlap_bits) - 1)
        if s0_top == s_last_bot == overlap:
            overlap_ok += 1

    base = float(np.mean(base_mses))
    tb = float(np.mean(tb_mses))
    gap = tb - base

    mse_ok = abs(gap) < 0.006
    prop_ok = overlap_ok == n_seqs
    print(f"  Base MSE:        {base:.4f}")
    print(f"  Tail-biting MSE: {tb:.4f}")
    print(f"  Gap:             {gap:+.4f}")
    print(f"  Overlap property: {overlap_ok}/{n_seqs}")
    print(f"  [{'PASS' if mse_ok else 'FAIL'}] MSE gap < 0.006")
    print(f"  [{'PASS' if prop_ok else 'FAIL'}] all tail-biting")
    return mse_ok and prop_ok


def main():
    print("=" * 60)
    print("Tripwire A.7.0: V-generic Viterbi (V=2 HYB)")
    print("=" * 60)

    results = []
    results.append(("A.7.0.1 backcompat", test_a700_backcompat()))
    results.append(("A.7.0.2 HYB V=2 MSE", test_a701_hyb_v2_table1()))
    results.append(("A.7.0.3 V=2 round-trip", test_a702_v2_roundtrip()))
    results.append(("A.7.0.4 V=2 tail-biting", test_a703_v2_tailbiting()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.0 GATE: PASS — V=2 Viterbi verified.")
        print("Ready for A.7.1 (BlockLDL decomposition).")
        sys.exit(0)
    else:
        print("A.7.0 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()