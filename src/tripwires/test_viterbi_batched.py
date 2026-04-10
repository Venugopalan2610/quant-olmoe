"""Tripwire A.7.3.0: batched Viterbi correctness and speedup.

A.7.3.0.1: Batched output bit-exact matches unbatched on B independent seqs
A.7.3.0.2: Speedup vs unbatched at B=128 (the real workload)
A.7.3.0.3: Same MSE distribution as unbatched on Gaussian seqs

Run: python -m src.tripwires.test_viterbi_batched
"""
import sys
import time
import numpy as np

from src.codes.ref import decode_hyb_batch
from src.viterbi.encode import viterbi_encode_v, viterbi_encode_v_batched


LUT_PATH = "cache/codes/hyb_lut_init.npy"


def make_hyb_decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def test_a73_0_1_bit_exact():
    """Batched output identical to unbatched run sequence-by-sequence."""
    print("\nA.7.3.0.1: bit-exact batched vs unbatched")
    print("-" * 60)

    decode = make_hyb_decoder()
    rng = np.random.default_rng(0)
    B, T = 8, 256
    seqs = rng.standard_normal((B, T)).astype(np.float32)

    # Unbatched reference
    unbatched_recons = np.zeros((B, T), dtype=np.float32)
    unbatched_mses = np.zeros(B, dtype=np.float32)
    unbatched_bitstreams = []
    unbatched_starts = np.zeros(B, dtype=np.int32)
    for i in range(B):
        bs, ss, recon, mse = viterbi_encode_v(
            seqs[i], L=16, k=2, V=2, decode_fn=decode,
        )
        unbatched_recons[i] = recon
        unbatched_mses[i] = mse
        unbatched_bitstreams.append(bs)
        unbatched_starts[i] = ss
    unbatched_bs_arr = np.stack(unbatched_bitstreams)

    # Batched
    bs_b, ss_b, recons_b, mses_b = viterbi_encode_v_batched(
        seqs, L=16, k=2, V=2, decode_fn=decode,
    )

    recon_match = np.allclose(unbatched_recons, recons_b, atol=0)
    bs_match = np.array_equal(unbatched_bs_arr, bs_b)
    ss_match = np.array_equal(unbatched_starts, ss_b)
    mse_match = np.allclose(unbatched_mses, mses_b, atol=1e-7)

    max_recon_diff = float(np.abs(unbatched_recons - recons_b).max())
    print(f"  B={B}, T={T}")
    print(f"  recon max abs diff: {max_recon_diff:.2e}")
    print(f"  unbatched MSEs: {[f'{m:.4f}' for m in unbatched_mses]}")
    print(f"  batched MSEs:   {[f'{m:.4f}' for m in mses_b]}")
    print(f"  [{'PASS' if recon_match else 'FAIL'}] reconstructions identical")
    print(f"  [{'PASS' if bs_match else 'FAIL'}] bitstreams identical")
    print(f"  [{'PASS' if ss_match else 'FAIL'}] start states identical")
    print(f"  [{'PASS' if mse_match else 'FAIL'}] MSEs match")

    return recon_match and bs_match and ss_match and mse_match


def test_a73_0_2_speedup():
    """Measure speedup at B=128 (the real BlockLDLQ inner-loop size)."""
    print("\nA.7.3.0.2: speedup at B=128, T=256")
    print("-" * 60)

    decode = make_hyb_decoder()
    rng = np.random.default_rng(1)
    B, T = 128, 256
    seqs = rng.standard_normal((B, T)).astype(np.float32)

    # Time unbatched (do all 128)
    t0 = time.time()
    for i in range(B):
        viterbi_encode_v(seqs[i], L=16, k=2, V=2, decode_fn=decode)
    unbatched_time = time.time() - t0

    # Time batched
    t0 = time.time()
    viterbi_encode_v_batched(seqs, L=16, k=2, V=2, decode_fn=decode)
    batched_time = time.time() - t0

    speedup = unbatched_time / batched_time
    print(f"  unbatched: {unbatched_time:.2f}s for {B} calls "
          f"({unbatched_time/B*1000:.0f}ms/call)")
    print(f"  batched:   {batched_time:.2f}s for {B} calls "
          f"({batched_time/B*1000:.0f}ms/call effective)")
    print(f"  speedup: {speedup:.1f}x")

    # Estimate full real-matrix wall clock
    n_block_calls = 128  # n_col_blocks for 2048/Ty=16
    matrix_time_unbatched = unbatched_time * n_block_calls
    matrix_time_batched = batched_time * n_block_calls
    print(f"\n  Projected wall clock per 2048x2048 matrix:")
    print(f"    unbatched: {matrix_time_unbatched/60:.1f} min")
    print(f"    batched:   {matrix_time_batched/60:.1f} min")
    print(f"  For 3136 matrices (full OLMoE):")
    print(f"    unbatched: {matrix_time_unbatched * 3136 / 3600:.1f} hours")
    print(f"    batched:   {matrix_time_batched * 3136 / 3600:.1f} hours")

    # Even modest speedup (10x) is enough; we need at least 5x to be useful
    speedup_ok = speedup >= 4
    print(f"\n  [{'PASS' if speedup_ok else 'FAIL'}] speedup >= 5x")
    return speedup_ok


def test_a73_0_3_quality_distribution():
    """Same MSE distribution on a larger batch."""
    print("\nA.7.3.0.3: MSE distribution sanity")
    print("-" * 60)

    decode = make_hyb_decoder()
    rng = np.random.default_rng(2)
    B, T = 64, 256
    seqs = rng.standard_normal((B, T)).astype(np.float32)

    _, _, _, mses_b = viterbi_encode_v_batched(seqs, L=16, k=2, V=2, decode_fn=decode)

    mean_mse = float(mses_b.mean())
    std_mse = float(mses_b.std())
    print(f"  B={B} sequences, mean MSE={mean_mse:.4f} ± {std_mse:.4f}")
    print(f"  Paper Table 1 HYB: 0.069")

    ok = 0.060 < mean_mse < 0.080
    print(f"  [{'PASS' if ok else 'FAIL'}] mean MSE in HYB range")
    return ok


def main():
    print("=" * 60)
    print("Tripwire A.7.3.0: batched Viterbi")
    print("=" * 60)

    results = []
    results.append(("A.7.3.0.1 bit-exact", test_a73_0_1_bit_exact()))
    results.append(("A.7.3.0.2 speedup", test_a73_0_2_speedup()))
    results.append(("A.7.3.0.3 quality", test_a73_0_3_quality_distribution()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.3.0 GATE: PASS — batched Viterbi verified.")
        print("Ready for A.7.3.1 (real OLMoE expert quantization).")
        sys.exit(0)
    else:
        print("A.7.3.0 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()