"""Tripwire A.5.1: embedding pass output sanity (sharded version).

Run: python -m src.tripwires.test_embed_pass
"""
import os
import sys
import numpy as np
import torch
from transformers import OlmoeForCausalLM

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
CALIB_PATH = "cache/calibration/tokens.npy"
SHARD_DIR = "cache/hidden_states/layer_00_input"


def load_meta():
    meta_path = os.path.join(SHARD_DIR, "meta.pt")
    if not os.path.exists(meta_path):
        return None
    return torch.load(meta_path, weights_only=True)


def load_shard(shard_idx):
    return torch.load(os.path.join(SHARD_DIR, f"shard_{shard_idx:04d}.pt"), weights_only=True)


def get_seq_from_shards(seq_idx, meta):
    """Read a single sequence by global index, loading only the relevant shard."""
    shard_idx = seq_idx // meta["shard_size"]
    in_shard_idx = seq_idx % meta["shard_size"]
    shard = load_shard(shard_idx)
    return shard[in_shard_idx]


def main():
    print("=" * 60)
    print("Tripwire A.5.1: embedding pass (sharded)")
    print("=" * 60)

    if not os.path.exists(SHARD_DIR):
        print(f"FAIL: {SHARD_DIR} does not exist.")
        print("Run: python -m src.hessian.embed_pass")
        sys.exit(1)

    meta = load_meta()
    if meta is None:
        print(f"FAIL: meta.pt missing in {SHARD_DIR}")
        sys.exit(1)

    print(f"\nMeta: {meta}")

    tokens = np.load(CALIB_PATH)
    n_seqs, seq_len = tokens.shape

    # A.5.1.1 — shape/dtype/shard count
    shape_ok = (
        meta["n_seqs"] == n_seqs
        and meta["seq_len"] == seq_len
        and meta["hidden_size"] == 2048
    )
    dtype_ok = meta["dtype"] == "bfloat16"
    expected_shards = (n_seqs + meta["shard_size"] - 1) // meta["shard_size"]
    n_shards_ok = meta["n_shards"] == expected_shards

    actual_shard_files = sorted(
        f for f in os.listdir(SHARD_DIR) if f.startswith("shard_")
    )
    files_ok = len(actual_shard_files) == expected_shards

    print(f"  [{'PASS' if shape_ok else 'FAIL'}] shapes match calibration ({n_seqs} seqs x {seq_len} tokens x 2048 hidden)")
    print(f"  [{'PASS' if dtype_ok else 'FAIL'}] dtype is bfloat16")
    print(f"  [{'PASS' if n_shards_ok else 'FAIL'}] shard count: meta says {meta['n_shards']}, expected {expected_shards}")
    print(f"  [{'PASS' if files_ok else 'FAIL'}] files on disk: {len(actual_shard_files)}, expected {expected_shards}")

    # A.5.1.2 — no NaN/Inf in any shard
    print(f"\nScanning all shards for NaN/Inf...")
    total_nan, total_inf = 0, 0
    for shard_idx in range(meta["n_shards"]):
        s = load_shard(shard_idx)
        total_nan += int(torch.isnan(s).sum())
        total_inf += int(torch.isinf(s).sum())
        del s
    finite_ok = (total_nan == 0 and total_inf == 0)
    print(f"  [{'PASS' if finite_ok else 'FAIL'}] no NaN ({total_nan}) or Inf ({total_inf})")

    # A.5.1.3 — bit-exact correctness on 4 random sequences
    print(f"\nLoading model for correctness check...")
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    embed = model.model.embed_tokens

    rng = np.random.default_rng(0)
    indices = rng.choice(n_seqs, size=4, replace=False).tolist()
    print(f"  Comparing cached vs fresh embedding for seqs {indices}")

    correct_count = 0
    with torch.no_grad():
        for idx in indices:
            cached_seq = get_seq_from_shards(idx, meta)
            seq_tokens = torch.from_numpy(tokens[idx].astype(np.int64)).unsqueeze(0)
            fresh = embed(seq_tokens).squeeze(0)
            if torch.equal(fresh, cached_seq):
                correct_count += 1
            else:
                max_diff = float((fresh.float() - cached_seq.float()).abs().max())
                print(f"  seq {idx}: MISMATCH (max abs diff {max_diff:.2e})")
    correctness_ok = correct_count == len(indices)
    print(f"  [{'PASS' if correctness_ok else 'FAIL'}] {correct_count}/{len(indices)} bit-exact")

    # A.5.1.4 — stats from shard 0
    s0 = load_shard(0).float()
    mean = float(s0.mean())
    std = float(s0.std())
    stats_ok = abs(mean) < 0.1 and 0.001 < std < 1.0
    print(f"\nShard 0 statistics:")
    print(f"  mean: {mean:+.6f}")
    print(f"  std:  {std:.6f}")
    print(f"  [{'PASS' if stats_ok else 'FAIL'}] stats in plausible range")

    print("\n" + "=" * 60)
    all_ok = shape_ok and dtype_ok and n_shards_ok and files_ok and finite_ok and correctness_ok and stats_ok
    if all_ok:
        print("A.5.1 GATE: PASS — sharded embedding cache verified.")
        print("Ready for A.5.2.")
        sys.exit(0)
    else:
        print("A.5.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()