"""A.5.4 — Collect Hessians for all OLMoE layers in a single pipeline.

Loads the model once on CPU in bf16. For each layer N in order:
  1. Read layer_{N}_input/ shards
  2. Move a fresh fp32 copy of layer N to GPU
  3. Forward all shards with expert-patched forward + attention collector
  4. Save layer N Hessians (4 attn + 128 expert files)
  5. Write layer_{N+1}_input/ shards (input to next layer)
  6. Delete layer_{N}_input/ (cleanup; skip with --keep-inputs)

Peak disk: ~60 GB total including model and accumulated Hessians.
Peak GPU: ~3 GB.
Peak CPU: ~15 GB.

Resume with --start-layer N if a previous run crashed at layer N.
"""
import os
import sys
import time
import shutil
import argparse
import gc
import copy
import torch
from transformers import OlmoeForCausalLM

from src.hessian.hooks import ExpertHessianCollector, install_expert_patch

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HIDDEN_DIR = "cache/hidden_states"
HESSIAN_DIR = "cache/hessians"


class AttentionHessianCollector:
    def __init__(self, layer, device):
        self.device = device
        self.linears = {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
        }
        self.hessians = {
            name: torch.zeros((lin.in_features, lin.in_features),
                              dtype=torch.float32, device=device)
            for name, lin in self.linears.items()
        }
        self.token_counts = {name: 0 for name in self.linears}
        self.handles = [
            lin.register_forward_pre_hook(self._make_hook(name))
            for name, lin in self.linears.items()
        ]

    def _make_hook(self, name):
        def hook(module, inputs):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).to(torch.float32)
            self.hessians[name].addmm_(x.T, x)
            self.token_counts[name] += x.shape[0]
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def save(self, layer_idx, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name in self.linears:
            n_tok = self.token_counts[name]
            H = self.hessians[name] / max(n_tok, 1)
            H = 0.5 * (H + H.T)
            torch.save({
                "H": H.cpu(), "n_tokens": n_tok,
                "layer": layer_idx, "kind": "attention", "proj": name,
            }, os.path.join(out_dir, f"attn_{name}.pt"))


def process_layer(model, rotary_emb, layer_idx, device, cfg):
    """Process a single layer end-to-end. Returns wall-clock seconds."""
    in_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
    out_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx + 1:02d}_input")
    hess_dir = os.path.join(HESSIAN_DIR, f"L{layer_idx:02d}")

    meta = torch.load(os.path.join(in_dir, "meta.pt"), weights_only=True)
    n_shards = meta["n_shards"]
    seq_len = meta["seq_len"]

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Fresh fp32 GPU copy of layer N (doesn't mutate the CPU model)
    layer = copy.deepcopy(model.model.layers[layer_idx]).to(
        device=device, dtype=torch.float32,
    )

    attn_collector = AttentionHessianCollector(layer, device)
    expert_collector = ExpertHessianCollector(
        num_experts=cfg.num_experts,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        device=device,
    )
    restore = install_expert_patch(layer.mlp.experts, expert_collector)

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    t_start = time.time()
    try:
        with torch.no_grad():
            for shard_idx in range(n_shards):
                shard_bf16 = torch.load(
                    os.path.join(in_dir, f"shard_{shard_idx:04d}.pt"),
                    weights_only=True,
                )
                B = shard_bf16.shape[0]
                shard_fp32 = shard_bf16.to(device=device, dtype=torch.float32)
                del shard_bf16

                pos_ids_batch = position_ids.expand(B, -1)
                cos, sin = rotary_emb(shard_fp32, pos_ids_batch)

                out = layer(
                    shard_fp32,
                    attention_mask=None,
                    position_ids=pos_ids_batch,
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )
                out_hidden = out[0] if isinstance(out, tuple) else out

                out_bf16 = out_hidden.to(dtype=torch.bfloat16).cpu()
                torch.save(out_bf16, os.path.join(out_dir, f"shard_{shard_idx:04d}.pt"))
                del shard_fp32, out_hidden, out_bf16, cos, sin
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if shard_idx % 8 == 0 or shard_idx == n_shards - 1:
                    elapsed = time.time() - t_start
                    print(f"    shard {shard_idx + 1}/{n_shards}  elapsed {elapsed:.1f}s")
    finally:
        restore()

    torch.save({**meta, "produced_by": f"layer_{layer_idx}"},
               os.path.join(out_dir, "meta.pt"))

    attn_collector.save(layer_idx, hess_dir)
    attn_collector.close()
    expert_collector.save(hess_dir, layer_idx)

    gu_counts = expert_collector.gate_up_counts.cpu()
    print(f"    expert load: min={int(gu_counts.min()):,} "
          f"max={int(gu_counts.max()):,} mean={int(gu_counts.float().mean()):,}")

    elapsed = time.time() - t_start

    del layer, attn_collector, expert_collector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument("--keep-inputs", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading full model (bf16 CPU)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    print(f"  loaded in {time.time() - t0:.1f}s")

    rotary_emb = copy.deepcopy(model.model.rotary_emb).to(
        device=device, dtype=torch.float32,
    )

    end_layer = args.end_layer if args.end_layer is not None else cfg.num_hidden_layers
    print(f"\nProcessing layers [{args.start_layer}, {end_layer})")

    grand_start = time.time()
    wall_clocks = []
    for layer_idx in range(args.start_layer, end_layer):
        in_dir = os.path.join(HIDDEN_DIR, f"layer_{layer_idx:02d}_input")
        if not os.path.exists(in_dir):
            print(f"\nFAIL: {in_dir} missing. Cannot process layer {layer_idx}.")
            sys.exit(1)

        print(f"\n=== Layer {layer_idx} ===")
        elapsed = process_layer(model, rotary_emb, layer_idx, device, cfg)
        wall_clocks.append(elapsed)
        print(f"  layer {layer_idx} done in {elapsed:.1f}s "
              f"(total elapsed {(time.time() - grand_start)/60:.1f} min)")

        # Clean up the just-consumed input cache
        if not args.keep_inputs:
            shutil.rmtree(in_dir)
            print(f"  cleaned up {in_dir}")

    total = time.time() - grand_start
    print(f"\n=== All {end_layer - args.start_layer} layers done ===")
    print(f"Total wall clock: {total:.1f}s ({total/60:.1f} min)")
    print(f"Per-layer avg: {sum(wall_clocks)/len(wall_clocks):.1f}s")

    torch.save({
        "wall_clocks": wall_clocks,
        "total_seconds": total,
        "layers_processed": list(range(args.start_layer, end_layer)),
    }, os.path.join(HESSIAN_DIR, "collection_stats.pt"))


if __name__ == "__main__":
    main()