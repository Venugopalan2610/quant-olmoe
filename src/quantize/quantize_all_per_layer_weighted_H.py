"""A.11 — Quantize all layers using per-LAYER mean Hessian instead of per-expert.

This is the routing-conditioned ablation: same trellis quantizer, same calibration
data, same everything — except the Hessian for each expert is the layer-mean H
across all 64 experts (i.e., routing is ignored at calibration time).

Output: cache/quantized_per_layer_H/L{NN}/{target}.pt

Compared against the per-expert baseline at cache/quantized/L{NN}/, the PPL gap
quantifies the value of routing-conditioned calibration.
"""
import os
import json
import time
import numpy as np
import torch
from safetensors import safe_open

from src.codes.ref import decode_hyb_batch
from src.rht.transform import make_sign_vector
from src.quantize.blockldlq import blockldlq
from src.quantize.serialize import save_quantized

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HESSIAN_DIR = "cache/hessians"                     # per-expert (existing)
HESSIAN_MEAN_DIR = "cache/hessians_per_layer_weighted" # per-layer mean (new)
QUANTIZED_DIR = "cache/quantized_per_layer_weighted_H"      # NEW output dir
LUT_PATH = "cache/codes/hyb_lut_init.npy"


def _make_decoder():
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def _safetensors_index():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)


def _load_weight(idx_json, key):
    shard = idx_json["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        return f.get_tensor(key).astype(np.float32)


def _load_attention_hessian(layer, proj):
    """Attention is unrouted — same Hessian as the per-expert baseline.

    The ablation is ONLY about expert routing. Attention quantization is
    held constant across configurations to isolate the routing effect.
    """
    path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"attn_{proj}.pt")
    return torch.load(path, weights_only=True)["H"].numpy()


def _load_layer_mean_hessian(layer, kind):
    """Load the per-layer mean Hessian (used for ALL experts in this layer)."""
    path = os.path.join(HESSIAN_MEAN_DIR, f"L{layer:02d}", f"layer_weighted_{kind}.pt")
    return torch.load(path, weights_only=True)["H"].numpy()


def quantize_layer_per_layer_H(layer_idx, num_experts=64, damp=0.01, verbose=True):
    """Quantize one layer using per-layer mean H for all experts."""
    out_dir = os.path.join(QUANTIZED_DIR, f"L{layer_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    idx_json = _safetensors_index()
    decode = _make_decoder()
    results = []

    layer_start = time.time()

    # === Attention projections (UNCHANGED — same as per-expert run) ===
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
        W = _load_weight(idx_json, key)
        H = _load_attention_hessian(layer_idx, proj)

        sign_l = make_sign_vector(W.shape[0], seed=layer_idx * 100000 + hash(proj) % 10000)
        sign_r = make_sign_vector(W.shape[1], seed=layer_idx * 100000 + hash(proj) % 10000 + 1)

        t0 = time.time()
        Wh, Wh_tilde, proxy, bs_dict = blockldlq(
            W, H, sign_l, sign_r, decode,
            use_cuda=True, return_bitstreams=True, damp=damp,
        )
        dt = time.time() - t0

        target_name = f"attn_{proj}"
        save_quantized(
            os.path.join(out_dir, f"{target_name}.pt"),
            bitstreams=bs_dict["bitstreams"],
            start_states=bs_dict["start_states"],
            sign_l=sign_l, sign_r=sign_r,
            W_scale=bs_dict["W_scale"],
            shape=W.shape,
            meta={"layer": layer_idx, "kind": "attention", "proj": proj,
                  "ablation": "per_layer_weighted_H"},
        )
        results.append({
            "name": target_name, "kind": "attention", "proj": proj,
            "expert": None, "shape": tuple(W.shape),
            "proxy_loss": proxy, "wall_time": dt,
        })
        if verbose:
            print(f"  L{layer_idx:02d} {target_name}: proxy={proxy:.4e} time={dt:.1f}s")

    # === Expert projections — USE LAYER MEAN H FOR ALL EXPERTS ===
    H_layer_gate_up = _load_layer_mean_hessian(layer_idx, "gate_up")
    H_layer_down    = _load_layer_mean_hessian(layer_idx, "down")

    for e in range(num_experts):
        for proj in ("gate_proj", "up_proj"):
            key = f"model.layers.{layer_idx}.mlp.experts.{e}.{proj}.weight"
            W = _load_weight(idx_json, key)

            sign_l = make_sign_vector(W.shape[0],
                                       seed=layer_idx * 100000 + e * 100 + hash(proj) % 100)
            sign_r = make_sign_vector(W.shape[1],
                                       seed=layer_idx * 100000 + e * 100 + hash(proj) % 100 + 1)

            t0 = time.time()
            Wh, Wh_tilde, proxy, bs_dict = blockldlq(
                W, H_layer_gate_up, sign_l, sign_r, decode,
                use_cuda=True, return_bitstreams=True, damp=damp,
            )
            dt = time.time() - t0

            target_name = f"expert_{e:02d}_{proj}"
            save_quantized(
                os.path.join(out_dir, f"{target_name}.pt"),
                bitstreams=bs_dict["bitstreams"],
                start_states=bs_dict["start_states"],
                sign_l=sign_l, sign_r=sign_r,
                W_scale=bs_dict["W_scale"],
                shape=W.shape,
                meta={"layer": layer_idx, "kind": "expert", "expert": e, "proj": proj,
                      "ablation": "per_layer_weighted_H"},
            )
            results.append({
                "name": target_name, "kind": "expert", "proj": proj,
                "expert": e, "shape": tuple(W.shape),
                "proxy_loss": proxy, "wall_time": dt,
            })

        # down_proj
        key = f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
        W = _load_weight(idx_json, key)

        sign_l = make_sign_vector(W.shape[0], seed=layer_idx * 100000 + e * 100 + 50)
        sign_r = make_sign_vector(W.shape[1], seed=layer_idx * 100000 + e * 100 + 51)

        t0 = time.time()
        Wh, Wh_tilde, proxy, bs_dict = blockldlq(
            W, H_layer_down, sign_l, sign_r, decode,
            use_cuda=True, return_bitstreams=True, damp=damp,
        )
        dt = time.time() - t0

        target_name = f"expert_{e:02d}_down_proj"
        save_quantized(
            os.path.join(out_dir, f"{target_name}.pt"),
            bitstreams=bs_dict["bitstreams"],
            start_states=bs_dict["start_states"],
            sign_l=sign_l, sign_r=sign_r,
            W_scale=bs_dict["W_scale"],
            shape=W.shape,
            meta={"layer": layer_idx, "kind": "expert", "expert": e, "proj": "down_proj",
                  "ablation": "per_layer_weighted_H"},
        )
        results.append({
            "name": target_name, "kind": "expert", "proj": "down_proj",
            "expert": e, "shape": tuple(W.shape),
            "proxy_loss": proxy, "wall_time": dt,
        })

        if verbose and (e + 1) % 16 == 0:
            elapsed = time.time() - layer_start
            print(f"  L{layer_idx:02d} progress: {e+1}/{num_experts} experts, "
                  f"{elapsed:.0f}s elapsed")

    layer_total = time.time() - layer_start
    if verbose:
        print(f"  L{layer_idx:02d} done: {len(results)} targets in {layer_total:.0f}s "
              f"({layer_total/len(results):.1f}s/target avg)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=16)
    parser.add_argument("--damp", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(QUANTIZED_DIR, exist_ok=True)
    print(f"A.11 routing ablation: per-layer mean H quantization")
    print(f"Output dir: {QUANTIZED_DIR}")
    print(f"Layers: [{args.start_layer}, {args.end_layer})")
    print(f"Damping: {args.damp}")

    grand_start = time.time()
    layer_times = []
    aggregate_proxy_per_expert = []
    aggregate_proxy_per_layer = []

    for layer_idx in range(args.start_layer, args.end_layer):
        # Resume support: skip if already complete
        out_dir = os.path.join(QUANTIZED_DIR, f"L{layer_idx:02d}")
        n_existing = len([f for f in os.listdir(out_dir)
                          if f.endswith(".pt")]) if os.path.exists(out_dir) else 0
        if n_existing >= 196:  # 4 attn + 64*3 expert
            print(f"\n=== Layer {layer_idx} === SKIP ({n_existing} files exist)")
            continue

        print(f"\n=== Layer {layer_idx} ===")
        results = quantize_layer_per_layer_H(layer_idx, damp=args.damp)
        layer_times.append(time.time() - grand_start)

        # Aggregate proxy losses for the headline comparison
        expert_losses = [r["proxy_loss"] for r in results if r["kind"] == "expert"]
        aggregate_proxy_per_layer.append(sum(expert_losses))
        print(f"  layer aggregate expert proxy loss: {sum(expert_losses):.4e}")

    total = time.time() - grand_start
    print(f"\n=== A.11 quantization done ===")
    print(f"Total wall clock: {total/60:.1f} min")

    # Save aggregate stats
    torch.save({
        "layer_aggregate_proxy": aggregate_proxy_per_layer,
        "total_seconds": total,
        "ablation": "per_layer_weighted_H",
    }, os.path.join(QUANTIZED_DIR, "ablation_stats.pt"))


if __name__ == "__main__":
    main()