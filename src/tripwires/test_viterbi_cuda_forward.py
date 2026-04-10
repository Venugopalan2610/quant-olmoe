"""Tripwire A.7.3.C.1: CUDA forward DP bit-exact vs numpy.

C.1.1: Single random sequence — final cum_err matches reference
C.1.2: 5 different seeds — all match within fp32 noise
C.1.3: Informational timing (no gate, just for sanity)

Run: python -m src.tripwires.test_viterbi_cuda_forward
"""
import sys
import time
import numpy as np

from src.codes.ref import decode_hyb_batch
from src.viterbi.encode import viterbi_encode_v
from src.cuda.viterbi_kernel import viterbi_forward_dp_cuda

LUT_PATH = "cache/codes/hyb_lut_init.npy"


def make_hyb_decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def test_c1_1_bit_exact():
    print("\nA.7.3.C.1.1: bit-exact cum_err vs numpy reference")
    print("-" * 60)

    decode = make_hyb_decoder()
    rng = np.random.default_rng(0)
    sequence = rng.standard_normal(256).astype(np.float32)

    # Numpy reference: extract trace_err[-1] = final cum_err
    _, _, _, _, trace, _ = viterbi_encode_v(
        sequence, L=16, k=2, V=2, decode_fn=decode, return_trace=True,
    )
    cum_err_numpy = trace[-1].astype(np.float32)

    cum_err_cuda = viterbi_forward_dp_cuda(
        sequence, L=16, k=2, V=2, decode_fn=decode,
    )

    max_abs = float(np.abs(cum_err_numpy - cum_err_cuda).max())
    rel = max_abs / max(float(np.abs(cum_err_numpy).max()), 1e-30)
    bit_exact = np.array_equal(cum_err_numpy, cum_err_cuda)

    print(f"  shape: {cum_err_numpy.shape}, dtype: {cum_err_cuda.dtype}")
    print(f"  numpy range: [{cum_err_numpy.min():.6f}, {cum_err_numpy.max():.6f}]")
    print(f"  cuda  range: [{cum_err_cuda.min():.6f}, {cum_err_cuda.max():.6f}]")
    print(f"  max abs diff: {max_abs:.2e}")
    print(f"  relative diff: {rel:.2e}")
    print(f"  bit-exact: {bit_exact}")

    ok = bit_exact or rel < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] match within tolerance (bit-exact or rel<1e-6)")
    return ok


def test_c1_2_multi_seed():
    print("\nA.7.3.C.1.2: 5 random seeds")
    print("-" * 60)

    decode = make_hyb_decoder()
    n_pass = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sequence = rng.standard_normal(256).astype(np.float32)

        _, _, _, _, trace, _ = viterbi_encode_v(
            sequence, L=16, k=2, V=2, decode_fn=decode, return_trace=True,
        )
        cum_err_numpy = trace[-1].astype(np.float32)
        cum_err_cuda = viterbi_forward_dp_cuda(
            sequence, L=16, k=2, V=2, decode_fn=decode,
        )

        max_abs = float(np.abs(cum_err_numpy - cum_err_cuda).max())
        rel = max_abs / max(float(np.abs(cum_err_numpy).max()), 1e-30)
        bit_exact = np.array_equal(cum_err_numpy, cum_err_cuda)
        ok = bit_exact or rel < 1e-6
        if ok:
            n_pass += 1
        marker = "BIT-EXACT" if bit_exact else f"rel={rel:.2e}"
        print(f"  seed {seed}: {marker} {'PASS' if ok else 'FAIL'}")

    print(f"  {n_pass}/5 PASS")
    return n_pass == 5


def test_c1_3_timing():
    print("\nA.7.3.C.1.3: informational timing")
    print("-" * 60)

    decode = make_hyb_decoder()
    rng = np.random.default_rng(0)
    sequence = rng.standard_normal(256).astype(np.float32)

    # Warmup (kernel compile + table upload)
    for _ in range(3):
        viterbi_forward_dp_cuda(sequence, L=16, k=2, V=2, decode_fn=decode)

    n_iter = 20
    t0 = time.time()
    for _ in range(n_iter):
        viterbi_forward_dp_cuda(sequence, L=16, k=2, V=2, decode_fn=decode)
    cuda_per = (time.time() - t0) / n_iter

    t0 = time.time()
    for _ in range(n_iter):
        viterbi_encode_v(sequence, L=16, k=2, V=2, decode_fn=decode)
    numpy_per = (time.time() - t0) / n_iter

    print(f"  numpy single-seq encode (full pipeline): {numpy_per*1000:.0f}ms")
    print(f"  cuda forward DP (incl table upload):     {cuda_per*1000:.0f}ms")
    print(f"  raw speedup: {numpy_per/cuda_per:.1f}x")
    print("  (informational only — C.2 measures proper batched speedup)")
    return True


def main():
    print("=" * 60)
    print("Tripwire A.7.3.C.1: single-seq Viterbi forward DP on CUDA")
    print("=" * 60)

    results = []
    results.append(("C.1.1 bit-exact",  test_c1_1_bit_exact()))
    results.append(("C.1.2 multi-seed", test_c1_2_multi_seed()))
    results.append(("C.1.3 timing",     test_c1_3_timing()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.3.C.1 GATE: PASS — CUDA forward DP verified.")
        print("Ready for A.7.3.C.2 (backtrace + batching).")
        sys.exit(0)
    else:
        print("A.7.3.C.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()