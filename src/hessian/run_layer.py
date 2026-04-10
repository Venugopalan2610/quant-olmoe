"""A.5.2+ — Run one transformer layer on cached hidden states.

With --with-experts, also monkey-patches OlmoeExperts.forward to collect
per-expert Hessians for gate_up and down inputs.
"""
import os
import sys
import time
import argparse
import shutil
import gc
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
        self.hessians = {}
        self.token_counts = {}
        for name, lin in self.linears.items():
            n = lin.in_features
            self.hessians[name] = torch.zeros((n, n), dtype=torch.float32, device=device)
            self.token_counts[name] = 0
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
            print(f"    saved {name}: n_tok={n_tok}, trace={float(H.trace()):.4f}")


def load_layer_and_rotary(model_dir, layer_idx, device):
    print(f"  Loading full model (bf16 CPU)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    layer = model.model.layers[layer_idx].to(dtype=torch.float32, device=device)
    rotary_emb = model.model.rotary_emb.to(dtype=torch.float32, device=device)
    cfg = model.config

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return layer, rotary_emb, cfg


def run_layer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    in_dir = os.path.join(HIDDEN_DIR, f"layer_{args.layer:02d}_input")
    out_dir = os.path.join(HIDDEN_DIR, f"layer_{args.layer + 1:02d}_input")
    hess_dir = os.path.join(HESSIAN_DIR, f"L{args.layer:02d}")

    if not os.path.exists(in_dir):
        print(f"FAIL: input shards missing at {in_dir}")
        sys.exit(1)

    meta = torch.load(os.path.join(in_dir, "meta.pt"), weights_only=True)
    print(f"Input meta: {meta}")

    if os.path.exists(out_dir):
        print(f"Removing existing {out_dir}...")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    layer, rotary_emb, cfg = load_layer_and_rotary(MODEL_DIR, args.layer, device)

    attn_collector = AttentionHessianCollector(layer, device)
    expert_collector = None
    restore_expert_patch = None

    if args.with_experts:
        expert_collector = ExpertHessianCollector(
            num_experts=cfg.num_experts,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            device=device,
        )
        restore_expert_patch = install_expert_patch(
            layer.mlp.experts, expert_collector,
        )
        print(f"  expert patch installed, collecting {cfg.num_experts} expert Hessians")

    seq_len = meta["seq_len"]
    n_shards = meta["n_shards"]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    print(f"Processing {n_shards} shards...")
    t0 = time.time()

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
                position_embeddings = (cos, sin)

                out = layer(
                    shard_fp32,
                    attention_mask=None,
                    position_ids=pos_ids_batch,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                out_hidden = out[0] if isinstance(out, tuple) else out

                out_bf16 = out_hidden.to(dtype=torch.bfloat16).cpu()
                torch.save(out_bf16, os.path.join(out_dir, f"shard_{shard_idx:04d}.pt"))
                del shard_fp32, out_hidden, out_bf16, cos, sin
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if shard_idx % 4 == 0 or shard_idx == n_shards - 1:
                    elapsed = time.time() - t0
                    pct = 100 * (shard_idx + 1) / n_shards
                    print(f"  shard {shard_idx + 1}/{n_shards} ({pct:.1f}%)  elapsed {elapsed:.1f}s")
    finally:
        if restore_expert_patch is not None:
            restore_expert_patch()

    torch.save({**meta, "produced_by": f"layer_{args.layer}"},
               os.path.join(out_dir, "meta.pt"))

    print(f"\nSaving attention Hessians to {hess_dir}/")
    attn_collector.save(args.layer, hess_dir)
    attn_collector.close()

    if expert_collector is not None:
        print(f"Saving {cfg.num_experts * 2} expert Hessians...")
        expert_collector.save(hess_dir, args.layer)
        # Summary stats
        gu_counts = expert_collector.gate_up_counts.cpu()
        print(f"  Expert gate_up token counts: "
              f"min={int(gu_counts.min())}, max={int(gu_counts.max())}, "
              f"mean={float(gu_counts.float().mean()):.0f}")

    print(f"\nDone in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--with-experts", action="store_true")
    args = parser.parse_args()
    run_layer(args)


if __name__ == "__main__":
    main()