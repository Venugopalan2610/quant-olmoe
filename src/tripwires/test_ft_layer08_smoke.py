"""Smoke test: collect activations for L08, FT all 64 experts + 4 attention projections."""
import os
import json
import time
import copy
import numpy as np
import torch
from safetensors import safe_open
from transformers import OlmoeForCausalLM

from src.quantize.serialize import load_quantized
from src.finetune.collect_activations import collect_layer_activations
from src.finetune.quant_expert import ft_one_expert, ft_one_linear_hweighted

LAYER = 8
MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
QUANT_DIR = f"cache/quantized/L{LAYER:02d}"
HESS_DIR = f"cache/hessians/L{LAYER:02d}"
LUT_PATH = "cache/codes/hyb_lut_init.npy"
DEVICE = "cuda:0"


def load_W(idx_json, key):
    shard = idx_json["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        return f.get_tensor(key).astype(np.float32)


def main():
    # Load model + rotary
    print("Loading model...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    rotary_emb = copy.deepcopy(model.model.rotary_emb).to(device=DEVICE, dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Collect activations for L08
    print(f"\nCollecting activations for layer {LAYER}...")
    captures = collect_layer_activations(
        model, rotary_emb, LAYER, cfg, DEVICE,
        budget_per_expert=1024,
    )

    # Get safetensors index
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        idx_json = json.load(f)

    lut_init = np.load(LUT_PATH)

    # ===== FT all 64 experts =====
    print(f"\nFine-tuning {cfg.num_experts} experts...")
    expert_results = []
    t_experts = time.time()
    for e in range(cfg.num_experts):
        X, Y = captures[e]
        if X is None:
            print(f"  expert {e}: SKIP (no tokens)")
            continue
        gate_p = load_quantized(f"{QUANT_DIR}/expert_{e:02d}_gate_proj.pt")
        up_p   = load_quantized(f"{QUANT_DIR}/expert_{e:02d}_up_proj.pt")
        down_p = load_quantized(f"{QUANT_DIR}/expert_{e:02d}_down_proj.pt")

        X_dev = X.to(DEVICE)
        Y_dev = Y.to(DEVICE)

        result = ft_one_expert(
            gate_p, up_p, down_p, lut_init,
            X_dev, Y_dev, n_steps=500, lr=5e-4, device=DEVICE,
        )
        expert_results.append({
            "expert": e,
            "n_tokens": X.shape[0],
            "loss_init": result["loss_init"],
            "loss_final": result["loss_final"],
            "improvement": result["improvement"],
        })
        if e % 8 == 0 or e == cfg.num_experts - 1:
            elapsed = time.time() - t_experts
            print(f"  expert {e:2d}: tokens={X.shape[0]:4d}  "
                  f"loss {result['loss_init']:.3e}->{result['loss_final']:.3e}  "
                  f"({result['improvement']*100:.0f}%)  elapsed {elapsed:.0f}s")
        del X_dev, Y_dev, result
        torch.cuda.empty_cache()

    t_expert_total = time.time() - t_experts
    avg_imp = np.mean([r["improvement"] for r in expert_results])
    print(f"\n  All experts FT: {t_expert_total:.0f}s, avg improvement {avg_imp*100:.1f}%")

    # ===== FT 4 attention projections (H-weighted) =====
    print(f"\nFine-tuning 4 attention projections (H-weighted)...")
    attn_results = []
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        W_key = f"model.layers.{LAYER}.self_attn.{proj}.weight"
        W_ref = load_W(idx_json, W_key)
        H = torch.load(f"{HESS_DIR}/attn_{proj}.pt", weights_only=True)["H"].numpy()
        payload = load_quantized(f"{QUANT_DIR}/attn_{proj}.pt")

        result = ft_one_linear_hweighted(
            payload, W_ref, H, lut_init,
            n_steps=500, lr=5e-4, damp=0.01, device=DEVICE,
        )
        attn_results.append({
            "proj": proj,
            "loss_init": result["loss_init"],
            "loss_final": result["loss_final"],
            "improvement": result["improvement"],
        })
        print(f"  {proj}: loss {result['loss_init']:.3e}->{result['loss_final']:.3e}  "
              f"({result['improvement']*100:.0f}%)")

    print("\n=== L08 SMOKE TEST DONE ===")
    print(f"  expert avg improvement: {avg_imp*100:.1f}%")
    attn_strs = [f"{r['improvement']*100:.0f}%" for r in attn_results]
    print(f"  attn improvements: {attn_strs}")

    if avg_imp > 0.1 and all(r["improvement"] > 0.05 for r in attn_results):
        print("[PASS] L08 smoke test")
    else:
        print("[INVESTIGATE] improvements lower than expected")


if __name__ == "__main__":
    main()