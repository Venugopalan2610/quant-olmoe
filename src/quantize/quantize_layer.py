"""Single-layer quantization driver.

Iterates over the ~132 QuantTargets in one OLMoE transformer layer
(4 attention + 64*3 expert projections), runs BlockLDLQ on each, saves
the quantized payload to cache/quantized/L{NN}/{target_name}.pt.

Loads weights directly from safetensors shards (no HF model load).
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
HESSIAN_DIR = "cache/hessians"
QUANTIZED_DIR = "cache/quantized"
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
    """Load one weight tensor from safetensors by full key."""
    shard = idx_json["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        return f.get_tensor(key).astype(np.float32)


def _load_attention_hessian(layer, proj):
    path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"attn_{proj}.pt")
    return torch.load(path, weights_only=True)["H"].numpy()


def _load_expert_hessian(layer, expert, kind):
    path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"expert_{expert:02d}_{kind}.pt")
    return torch.load(path, weights_only=True)["H"].numpy()


def quantize_layer(layer_idx, num_experts=64, damp=0.01, verbose=True):
    """Quantize all 132 targets in one OLMoE layer.

    Returns:
        results: list of dicts, one per target, with keys:
            name, kind, proj, expert, shape, proxy_loss, wall_time
    """
    out_dir = os.path.join(QUANTIZED_DIR, f"L{layer_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    idx_json = _safetensors_index()
    decode = _make_decoder()
    results = []

    layer_start = time.time()

    # === Attention projections ===
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
            meta={"layer": layer_idx, "kind": "attention", "proj": proj},
        )

        results.append({
            "name": target_name, "kind": "attention", "proj": proj,
            "expert": None, "shape": tuple(W.shape),
            "proxy_loss": proxy, "wall_time": dt,
        })
        if verbose:
            print(f"  L{layer_idx:02d} {target_name}: shape={W.shape} "
                  f"proxy={proxy:.4e} time={dt:.1f}s")

    # === Expert projections ===
    for e in range(num_experts):
        # gate_proj and up_proj share gate_up_H
        H_gate_up = _load_expert_hessian(layer_idx, e, "gate_up")

        for proj in ("gate_proj", "up_proj"):
            key = f"model.layers.{layer_idx}.mlp.experts.{e}.{proj}.weight"
            W = _load_weight(idx_json, key)

            sign_l = make_sign_vector(W.shape[0],
                                       seed=layer_idx * 100000 + e * 100 + hash(proj) % 100)
            sign_r = make_sign_vector(W.shape[1],
                                       seed=layer_idx * 100000 + e * 100 + hash(proj) % 100 + 1)

            t0 = time.time()
            Wh, Wh_tilde, proxy, bs_dict = blockldlq(
                W, H_gate_up, sign_l, sign_r, decode,
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
                meta={"layer": layer_idx, "kind": "expert", "expert": e, "proj": proj},
            )
            results.append({
                "name": target_name, "kind": "expert", "proj": proj,
                "expert": e, "shape": tuple(W.shape),
                "proxy_loss": proxy, "wall_time": dt,
            })

        # down_proj
        H_down = _load_expert_hessian(layer_idx, e, "down")
        key = f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
        W = _load_weight(idx_json, key)

        sign_l = make_sign_vector(W.shape[0], seed=layer_idx * 100000 + e * 100 + 50)
        sign_r = make_sign_vector(W.shape[1], seed=layer_idx * 100000 + e * 100 + 51)

        t0 = time.time()
        Wh, Wh_tilde, proxy, bs_dict = blockldlq(
            W, H_down, sign_l, sign_r, decode,
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
            meta={"layer": layer_idx, "kind": "expert", "expert": e, "proj": "down_proj"},
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