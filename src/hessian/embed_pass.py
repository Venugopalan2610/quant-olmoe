"""A.5.1 — Embedding pass (sharded).

Loads OLMoE on CPU, runs the embedding layer on all calibration sequences
in batches, saves the resulting hidden states to disk as multiple shards.

Output: cache/hidden_states/layer_00_input/shard_NNNN.pt
        cache/hidden_states/layer_00_input/meta.pt

Sharding keeps peak memory bounded — we never hold more than one shard
plus the model in RAM at once. Critical for WSL2 which OOMs hard rather
than swapping gracefully.
"""
import os
import sys
import time
import shutil
import argparse
import numpy as np
import torch
from transformers import OlmoeForCausalLM

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
CALIB_PATH = "cache/calibration/tokens.npy"
HIDDEN_DIR = "cache/hidden_states"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_size", type=int, default=64,
                        help="Sequences per shard. 64 seqs * 1024 tokens * 2048 hidden * 2 bytes "
                             "= 268 MB per shard, well within memory budget.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    out_dir = os.path.join(HIDDEN_DIR, "layer_00_input")
    if os.path.exists(out_dir):
        print(f"Removing existing {out_dir}...")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    tokens = np.load(CALIB_PATH)
    n_seqs, seq_len = tokens.shape
    print(f"Calibration: {n_seqs} sequences x {seq_len} tokens")

    n_shards = (n_seqs + args.shard_size - 1) // args.shard_size
    print(f"Sharding: {n_shards} shards of up to {args.shard_size} sequences each")

    print(f"Loading OLMoE in bf16...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    embed = model.model.embed_tokens
    hidden_size = model.config.hidden_size

    print(f"\nRunning embedding pass...")
    t0 = time.time()
    total_nan, total_inf = 0, 0
    sum_for_stats = torch.zeros((), dtype=torch.float64)
    sumsq_for_stats = torch.zeros((), dtype=torch.float64)
    n_for_stats = 0

    with torch.no_grad():
        for shard_idx in range(n_shards):
            i = shard_idx * args.shard_size
            j = min(i + args.shard_size, n_seqs)
            n_in_shard = j - i

            shard = torch.zeros((n_in_shard, seq_len, hidden_size), dtype=torch.bfloat16)

            for b_start in range(0, n_in_shard, args.batch_size):
                b_end = min(b_start + args.batch_size, n_in_shard)
                batch_tokens = torch.from_numpy(tokens[i + b_start:i + b_end].astype(np.int64))
                shard[b_start:b_end] = embed(batch_tokens)

            # Sanity check this shard
            n_nan = int(torch.isnan(shard).sum())
            n_inf = int(torch.isinf(shard).sum())
            total_nan += n_nan
            total_inf += n_inf

            # Accumulate stats from first 4 shards only (cheap)
            if shard_idx < 4:
                sf = shard.float()
                sum_for_stats += sf.sum()
                sumsq_for_stats += (sf ** 2).sum()
                n_for_stats += sf.numel()

            shard_path = os.path.join(out_dir, f"shard_{shard_idx:04d}.pt")
            torch.save(shard, shard_path)
            del shard

            if shard_idx % 4 == 0 or shard_idx == n_shards - 1:
                pct = 100 * (shard_idx + 1) / n_shards
                elapsed = time.time() - t0
                print(f"  shard {shard_idx + 1}/{n_shards} ({pct:.1f}%)  elapsed {elapsed:.1f}s")

    if total_nan or total_inf:
        print(f"FAIL: embeddings contain NaN ({total_nan}) or Inf ({total_inf})")
        sys.exit(1)

    # Save meta
    meta = {
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "shard_size": args.shard_size,
        "n_shards": n_shards,
        "dtype": "bfloat16",
    }
    torch.save(meta, os.path.join(out_dir, "meta.pt"))

    mean = float(sum_for_stats / n_for_stats)
    var = float(sumsq_for_stats / n_for_stats) - mean ** 2
    std = float(var ** 0.5)
    print(f"\nFirst 4 shards statistics:")
    print(f"  mean: {mean:+.6f}")
    print(f"  std:  {std:.6f}")

    total_size = sum(
        os.path.getsize(os.path.join(out_dir, f))
        for f in os.listdir(out_dir)
    )
    print(f"\nTotal shard size on disk: {total_size / 1e9:.2f} GB")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()