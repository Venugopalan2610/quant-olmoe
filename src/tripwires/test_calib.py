"""Tripwire A.5.0: calibration data sanity.

A.5.0.1: Output file exists with target shape and dtype
A.5.0.2: Token IDs in valid range [0, vocab_size)
A.5.0.3: Decoded samples look like real text
A.5.0.4: No constant/repetitive sequences (degenerate calibration)

Run: python -m src.tripwires.test_calib
"""
import os
import sys
import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
CALIB_PATH = "cache/calibration/tokens.npy"


def main():
    print("=" * 60)
    print("Tripwire A.5.0: calibration data")
    print("=" * 60)

    if not os.path.exists(CALIB_PATH):
        print(f"FAIL: {CALIB_PATH} does not exist.")
        print("Run: python -m src.hessian.prepare_calib")
        sys.exit(1)

    arr = np.load(CALIB_PATH)
    print(f"\nLoaded {CALIB_PATH}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  size on disk: {os.path.getsize(CALIB_PATH) / 1e6:.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # A.5.0.1: shape and dtype
    shape_ok = arr.ndim == 2 and arr.shape[0] >= 1024 and arr.shape[1] >= 512
    dtype_ok = arr.dtype in (np.int32, np.int64)
    print(f"\n  [{'PASS' if shape_ok else 'FAIL'}] shape sane (>=1024 seqs, >=512 tokens)")
    print(f"  [{'PASS' if dtype_ok else 'FAIL'}] dtype is integer")

    # A.5.0.2: token IDs in valid range
    in_range = (arr.min() >= 0) and (arr.max() < tokenizer.vocab_size)
    print(f"  [{'PASS' if in_range else 'FAIL'}] all token IDs in [0, {tokenizer.vocab_size})")
    print(f"    actual range: [{arr.min()}, {arr.max()}]")

    # A.5.0.3: decoded samples look like text
    print(f"\n  Sample sequence 0 (first 80 tokens):")
    sample = tokenizer.decode(arr[0, :80].tolist())
    print(f"    {sample!r}")
    print(f"  Sample sequence {arr.shape[0]//2} (first 80 tokens):")
    sample_mid = tokenizer.decode(arr[arr.shape[0]//2, :80].tolist())
    print(f"    {sample_mid!r}")

    # Heuristic: real text has spaces and a mix of common words
    has_text_0 = " " in sample and len(set(sample.split())) >= 5
    has_text_mid = " " in sample_mid and len(set(sample_mid.split())) >= 5
    text_ok = has_text_0 and has_text_mid
    print(f"  [{'PASS' if text_ok else 'FAIL'}] decoded samples look like real text")

    # A.5.0.4: no degenerate sequences (entire sequence is one repeated token)
    n_unique_per_seq = np.array([len(set(row.tolist())) for row in arr[:50]])
    diversity_ok = (n_unique_per_seq > 50).all()
    print(f"  [{'PASS' if diversity_ok else 'FAIL'}] no degenerate sequences "
          f"(min unique tokens in first 50 seqs: {n_unique_per_seq.min()})")

    print("\n" + "=" * 60)
    all_ok = shape_ok and dtype_ok and in_range and text_ok and diversity_ok
    if all_ok:
        print("A.5.0 GATE: PASS — calibration data verified.")
        print("Ready for A.5.1 (embedding pass).")
        sys.exit(0)
    else:
        print("A.5.0 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()