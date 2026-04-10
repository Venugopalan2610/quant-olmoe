"""Prepare calibration data: stream Dolma, tokenize, pack into fixed-length
sequences, save as a single numpy file.

Output: cache/calibration/tokens.npy of shape (N_SEQS, SEQ_LEN), int32

Strategy:
- Stream Dolma to avoid downloading the full dataset
- Concatenate documents with EOS separators until we have N_SEQS * SEQ_LEN tokens
- Reshape into fixed sequences

This is "packed" calibration data, which is standard for PTQ — we don't care
about preserving document boundaries because the Hessian is just an average
of input outer products.
"""
import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
CALIB_DIR = "cache/calibration"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seqs", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_docs", type=int, default=200_000,
                        help="Max documents to scan; should be plenty for the target token count")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(CALIB_DIR, exist_ok=True)
    out_path = os.path.join(CALIB_DIR, "tokens.npy")
    meta_path = os.path.join(CALIB_DIR, "meta.txt")

    if os.path.exists(out_path):
        existing = np.load(out_path)
        if existing.shape == (args.n_seqs, args.seq_len):
            print(f"Calibration data already exists at {out_path} with target shape, skipping.")
            print(f"Delete the file to regenerate.")
            return

    target_tokens = args.n_seqs * args.seq_len
    print(f"Target: {args.n_seqs} sequences x {args.seq_len} tokens = {target_tokens:,} tokens")

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    eos_id = tokenizer.eos_token_id
    print(f"EOS token id: {eos_id}, vocab size: {tokenizer.vocab_size}")

    print("Streaming Dolma (this will not download the full dataset)...")
    # The default Dolma config has many subsets; we want a representative slice
    # Use the 'v1_7' config which is the standard, and stream
    try:
        ds = load_dataset(
            "allenai/dolma",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"Failed to load Dolma: {e}")
        print("Falling back to allenai/c4 as a substitute (still reasonable for OLMoE)")
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    rng = np.random.default_rng(args.seed)

    buffer = []
    buffer_len = 0
    docs_seen = 0
    sources_seen = {}

    for doc in ds:
        docs_seen += 1
        if docs_seen > args.max_docs:
            print(f"Hit max_docs limit ({args.max_docs}) before reaching target tokens")
            break

        text = doc.get("text", "")
        if not text or len(text) < 50:
            continue

        source = doc.get("source", "unknown")
        sources_seen[source] = sources_seen.get(source, 0) + 1

        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)
        buffer.append(eos_id)
        buffer_len += len(ids) + 1

        if buffer_len >= target_tokens:
            break

        if docs_seen % 1000 == 0:
            pct = 100 * buffer_len / target_tokens
            print(f"  docs seen: {docs_seen:,}  buffer: {buffer_len:,} tokens ({pct:.1f}%)")

    if buffer_len < target_tokens:
        print(f"WARN: only collected {buffer_len:,} / {target_tokens:,} tokens. "
              f"Will pad by tiling.")
        # Tile to reach target
        while buffer_len < target_tokens:
            buffer.extend(buffer[:target_tokens - buffer_len])
            buffer_len = len(buffer)

    arr = np.array(buffer[:target_tokens], dtype=np.int32)
    arr = arr.reshape(args.n_seqs, args.seq_len)

    print(f"\nFinal shape: {arr.shape}")
    print(f"Token range: [{arr.min()}, {arr.max()}]")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Documents scanned: {docs_seen:,}")
    print(f"Source distribution (top 5):")
    for src, count in sorted(sources_seen.items(), key=lambda x: -x[1])[:5]:
        print(f"  {src}: {count:,} docs")

    np.save(out_path, arr)
    print(f"\nSaved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.2f} MB")

    # Write a small meta file
    with open(meta_path, "w") as f:
        f.write(f"shape: {arr.shape}\n")
        f.write(f"dtype: {arr.dtype}\n")
        f.write(f"docs_scanned: {docs_seen}\n")
        f.write(f"sources:\n")
        for src, count in sorted(sources_seen.items(), key=lambda x: -x[1]):
            f.write(f"  {src}: {count}\n")


if __name__ == "__main__":
    main()