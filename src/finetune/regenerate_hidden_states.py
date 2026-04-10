"""Regenerate cache/hidden_states/layer_{N}_input/ for activation collection.

RESUMABLE: skips layers whose output dir is already complete (all shards present).
For partially-complete layers, resumes from the last missing shard.

The Hessian collection pipeline deleted these to save disk. We need them back
to feed into collect_layer_activations. Replays embed → layer 0 → layer 1 → ...
through the full fp16 model, dumping shard tensors at each layer boundary.

Reuses cache/calibration/tokens.npy as the input.
"""
import os
import time
import shutil
import gc
import copy
import argparse
import numpy as np
import torch
from transformers import OlmoeForCausalLM

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HIDDEN_DIR = "cache/hidden_states"
TOKENS_PATH = "cache/calibration/tokens.npy"


def _count_shards(directory):
    """Count shard_*.pt files in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.startswith("shard_") and f.endswith(".pt")])


def _layer_complete(directory, expected_shards):
    """Check if a layer dir has all expected shards + meta."""
    if not os.path.exists(directory):
        return False
    if not os.path.exists(os.path.join(directory, "meta.pt")):
        return False
    return _count_shards(directory) >= expected_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-batch-size", type=int, default=8,
                        help="sequences per shard")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokens...")
    tokens = np.load(TOKENS_PATH)
    print(f"  tokens shape: {tokens.shape}, dtype: {tokens.dtype}")

    print("Loading full model (bf16 CPU)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    print(f"  loaded in {time.time() - t0:.1f}s")

    end_layer = args.end_layer if args.end_layer is not None else cfg.num_hidden_layers

    n_seqs, seq_len = tokens.shape
    B = args.shard_batch_size
    n_shards = (n_seqs + B - 1) // B
    print(f"  {n_seqs} seqs x {seq_len} tokens = {n_shards} shards of B={B}")

    # ====== Step 1: embed and write layer_00_input (skip if complete) ======
    in_dir = os.path.join(HIDDEN_DIR, "layer_00_input")
    if _layer_complete(in_dir, n_shards):
        print(f"\nlayer_00_input already complete ({_count_shards(in_dir)} shards), skipping embed")
    else:
        os.makedirs(in_dir, exist_ok=True)
        embed = copy.deepcopy(model.model.embed_tokens).to(device=device, dtype=torch.float32)
        print(f"\nEmbedding {n_shards} shards...")
        t0 = time.time()
        with torch.no_grad():
            for s in range(n_shards):
                out_path = os.path.join(in_dir, f"shard_{s:04d}.pt")
                if os.path.exists(out_path):
                    continue
                start = s * B
                end = min(start + B, n_seqs)
                shard_tokens = torch.from_numpy(tokens[start:end]).long().to(device)
                shard_embeds = embed(shard_tokens).to(torch.bfloat16).cpu()
                torch.save(shard_embeds, out_path)
                del shard_tokens, shard_embeds
        torch.save(
            {"n_shards": n_shards, "seq_len": seq_len, "shard_batch_size": B},
            os.path.join(in_dir, "meta.pt"),
        )
        del embed
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  embedded in {time.time() - t0:.1f}s")

    # ====== Step 2: replay through layers ======
    rotary_emb = copy.deepcopy(model.model.rotary_emb).to(device=device, dtype=torch.float32)

    for layer_idx in range(args.start_layer, end_layer):
        cur_in_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
        out_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx + 1:02d}_input")

        if not os.path.exists(cur_in_dir):
            print(f"\nFAIL: {cur_in_dir} missing. Cannot process layer {layer_idx}.")
            return

        meta = torch.load(os.path.join(cur_in_dir, "meta.pt"), weights_only=True)
        n_shards_l = meta["n_shards"]

        # Skip if output is already complete
        if _layer_complete(out_dir, n_shards_l):
            print(f"\n=== Layer {layer_idx} === SKIP (output complete: {_count_shards(out_dir)} shards)")
            continue

        os.makedirs(out_dir, exist_ok=True)
        existing = _count_shards(out_dir)
        print(f"\n=== Layer {layer_idx} === resuming from shard {existing}/{n_shards_l}")

        seq_len_l = meta["seq_len"]
        layer = copy.deepcopy(model.model.layers[layer_idx]).to(
            device=device, dtype=torch.float32,
        )
        position_ids = torch.arange(seq_len_l, dtype=torch.long, device=device).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            for s in range(n_shards_l):
                out_path = os.path.join(out_dir, f"shard_{s:04d}.pt")
                if os.path.exists(out_path):
                    continue  # already done

                shard_bf16 = torch.load(
                    os.path.join(cur_in_dir, f"shard_{s:04d}.pt"), weights_only=True,
                )
                Bs = shard_bf16.shape[0]
                shard_fp32 = shard_bf16.to(device=device, dtype=torch.float32)
                del shard_bf16

                pos_ids = position_ids.expand(Bs, -1)
                cos, sin = rotary_emb(shard_fp32, pos_ids)
                out = layer(
                    shard_fp32, attention_mask=None,
                    position_ids=pos_ids, position_embeddings=(cos, sin),
                    use_cache=False,
                )
                out_hidden = out[0] if isinstance(out, tuple) else out
                torch.save(
                    out_hidden.to(torch.bfloat16).cpu(),
                    out_path,
                )
                del shard_fp32, out_hidden, out, cos, sin, pos_ids
                torch.cuda.empty_cache()

                if s % 16 == 0 or s == n_shards_l - 1:
                    elapsed = time.time() - t0
                    print(f"    shard {s + 1}/{n_shards_l}  elapsed {elapsed:.1f}s")

        torch.save(meta, os.path.join(out_dir, "meta.pt"))

        # Free GPU memory aggressively between layers
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  layer {layer_idx} done in {time.time() - t0:.1f}s")

    print(f"\n=== All layers regenerated ===")
    # Summary
    for layer_idx in range(end_layer + 1):
        d = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
        n = _count_shards(d) if os.path.exists(d) else 0
        print(f"  layer_{layer_idx:02d}_input: {n} shards")


if __name__ == "__main__":
    main()