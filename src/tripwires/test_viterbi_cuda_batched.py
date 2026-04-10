"""Tripwire A.7.3.C.2: batched CUDA Viterbi with backtrace.

A.7.3.C.2.1: Bit-exact (or near) match against numpy batched encoder
A.7.3.C.2.2: Speedup measurement at B=128, T=256

Run: python -m src.tripwires.test_viterbi_cuda_batched
"""
import sys
import time
import numpy as np
import torch

from src.codes.ref import decode_hyb_batch
from src.viterbi.encode import (
    viterbi_encode_v_batched,
    precompute_codebook_v,
)
from src.viterbi.cuda_kernel import viterbi_encode_v_batched_cuda


LUT_PATH = "cache/codes/hyb_lut_init.npy"


def _decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def test_correctness():
    print("\nA.7.3.C.2.1: batched correctness vs numpy")
    print("-" * 60)

    decode = _decoder()
    rng = np.random.default_rng(0)
    B, T = 32, 256
    seqs = rng.standard_normal((B, T)).astype(np.float32)

    # Numpy reference
    bs_np, ss_np, recons_np, mses_np = viterbi_encode_v_batched(
        seqs, L=16, k=2, V=2, decode_fn=decode,
    )

    # CUDA
    codebook_np = precompute_codebook_v(L=16, V=2, decode_fn=decode)
    seqs_gpu = torch.from_numpy(seqs).cuda()
    cb_gpu = torch.from_numpy(codebook_np).cuda().contiguous()

    print("  warming kernel (first call may JIT-compile, ~30-60s)...")
    _ = viterbi_encode_v_batched_cuda(seqs_gpu, cb_gpu)
    torch.cuda.synchronize()
    print("  warm done")

    bs_cu, ss_cu, recons_cu, mses_cu = viterbi_encode_v_batched_cuda(seqs_gpu, cb_gpu)
    bs_cu_np = bs_cu.cpu().numpy()
    ss_cu_np = ss_cu.cpu().numpy()
    recons_cu_np = recons_cu.cpu().numpy()
    mses_cu_np = mses_cu.cpu().numpy()

    bs_match = np.array_equal(bs_np, bs_cu_np)
    ss_match = np.array_equal(ss_np, ss_cu_np)
    recon_max_diff = float(np.abs(recons_np - recons_cu_np).max())
    mse_max_diff = float(np.abs(mses_np - mses_cu_np).max())

    print(f"  B={B}, T={T}")
    print(f"  bitstreams match: {bs_match}")
    print(f"  start_states match: {ss_match}")
    print(f"  recon max abs diff: {recon_max_diff:.2e}")
    print(f"  mse max abs diff:   {mse_max_diff:.2e}")
    print(f"  numpy MSEs (first 4): {mses_np[:4]}")
    print(f"  cuda  MSEs (first 4): {mses_cu_np[:4]}")

    # Bit-exact would be ideal but fp noise allowed: rel < 1e-5
    bs_ok = bs_match
    ss_ok = ss_match
    recon_ok = recon_max_diff < 1e-4
    mse_ok = mse_max_diff < 1e-5

    print(f"  [{'PASS' if bs_ok else 'FAIL'}] bitstreams identical")
    print(f"  [{'PASS' if ss_ok else 'FAIL'}] start_states identical")
    print(f"  [{'PASS' if recon_ok else 'FAIL'}] recons close (diff < 1e-4)")
    print(f"  [{'PASS' if mse_ok else 'FAIL'}] mses close")

    return bs_ok and ss_ok and recon_ok and mse_ok


def test_speedup():
    print("\nA.7.3.C.2.2: speedup at B=128, T=256")
    print("-" * 60)

    decode = _decoder()
    rng = np.random.default_rng(1)
    B, T = 128, 256
    seqs = rng.standard_normal((B, T)).astype(np.float32)

    # Numpy batched timing
    n_iter = 3
    t0 = time.time()
    for _ in range(n_iter):
        viterbi_encode_v_batched(seqs, L=16, k=2, V=2, decode_fn=decode)
    numpy_time = (time.time() - t0) / n_iter

    # CUDA batched timing
    codebook_np = precompute_codebook_v(L=16, V=2, decode_fn=decode)
    seqs_gpu = torch.from_numpy(seqs).cuda()
    cb_gpu = torch.from_numpy(codebook_np).cuda().contiguous()

    # Warmup
    for _ in range(2):
        viterbi_encode_v_batched_cuda(seqs_gpu, cb_gpu)
        torch.cuda.synchronize()

    n_iter_cuda = 5
    t0 = time.time()
    for _ in range(n_iter_cuda):
        viterbi_encode_v_batched_cuda(seqs_gpu, cb_gpu)
        torch.cuda.synchronize()
    cuda_time = (time.time() - t0) / n_iter_cuda

    speedup = numpy_time / cuda_time
    print(f"  numpy batched: {numpy_time*1000:.0f} ms/call")
    print(f"  cuda  batched: {cuda_time*1000:.0f} ms/call")
    print(f"  speedup: {speedup:.0f}x")

    n_block_calls = 128  # n_col_blocks for 2048-dim weight matrix
    matrix_min = (cuda_time * n_block_calls) / 60
    full_hours = matrix_min * 3136 / 60
    print(f"\n  Projected per 2048x2048 matrix: {matrix_min:.1f} min")
    print(f"  Projected full OLMoE (3136 mat):  {full_hours:.1f} hours")

    ok = speedup >= 50
    print(f"\n  [{'PASS' if ok else 'FAIL'}] speedup >= 50x")
    return ok


def main():
    print("=" * 60)
    print("Tripwire A.7.3.C.2: batched CUDA Viterbi")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    results = []
    results.append(("correctness", test_correctness()))
    results.append(("speedup", test_speedup()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.3.C.2 GATE: PASS — batched CUDA encoder works.")
        print("Ready for A.7.3.C.3 (BlockLDLQ + real expert).")
        sys.exit(0)
    else:
        print("A.7.3.C.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()