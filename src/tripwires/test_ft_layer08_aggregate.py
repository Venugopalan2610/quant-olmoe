"""Layer 8 full FT with aggregate stats. Saves tuned LUTs but does not install.

Verdict criteria:
  - aggregate forward MSE improvement > 10% → FT is real, run partial PPL
  - aggregate forward MSE improvement < 5%  → FT is structural, pivot to no-FT story
"""
import os
import json
import time
import copy
import gc
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
QUANT_FT_DIR = f"cache/quantized_ft/L{LAYER:02d}"
DEVICE = "cuda:0"

# Use lr=2e-3, which converged fastest in the LR sweep
LR_EXPERT = 2e-3
LR_ATTN = 5e-4  # H-weighted is a different loss surface, use slower lr
N_STEPS = 300   # 500 was overkill, all converged by ~100


def load_W(idx_json, key):
    shard = idx_json["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        return f.get_tensor(key).astype(np.float32)


def main():
    os.makedirs(QUANT_FT_DIR, exist_ok=True)

    print("Loading model...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    rotary_emb = copy.deepcopy(model.model.rotary_emb).to(device=DEVICE, dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.0f}s")

    # Collect activations
    print(f"\nCollecting activations for layer {LAYER}...")
    captures = collect_layer_activations(
        model, rotary_emb, LAYER, cfg, DEVICE, budget_per_expert=1024,
    )

    # Get safetensors index BEFORE freeing the model dir reference
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        idx_json = json.load(f)

    # Free the model — only needed for activation collection
    del model, rotary_emb
    gc.collect()
    torch.cuda.empty_cache()

    lut_init = np.load(LUT_PATH)

    # ===== FT all 64 experts =====
    print(f"\nFine-tuning {cfg.num_experts} experts (lr={LR_EXPERT}, n_steps={N_STEPS})...")
    expert_results = []
    layer_luts = {}  # target_name -> tuned LUT
    t_experts = time.time()

    sum_init = 0.0
    sum_final = 0.0
    sum_tokens = 0

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
            X_dev, Y_dev, n_steps=N_STEPS, lr=LR_EXPERT, device=DEVICE,
        )

        n = X.shape[0]
        sum_init  += result["loss_init"]  * n
        sum_final += result["loss_final"] * n
        sum_tokens += n

        expert_results.append({
            "expert": e, "n_tokens": n,
            "loss_init": result["loss_init"],
            "loss_final": result["loss_final"],
            "improvement": result["improvement"],
        })
        layer_luts[f"expert_{e:02d}_gate_proj"] = result["lut_gate"]
        layer_luts[f"expert_{e:02d}_up_proj"]   = result["lut_up"]
        layer_luts[f"expert_{e:02d}_down_proj"] = result["lut_down"]

        if e % 8 == 0 or e == cfg.num_experts - 1:
            elapsed = time.time() - t_experts
            print(f"  expert {e:2d}: {result['improvement']*100:.1f}%  elapsed {elapsed:.0f}s")

        del X_dev, Y_dev, result
        torch.cuda.empty_cache()

    # ===== FT 4 attention projections (H-weighted) =====
    print(f"\nFine-tuning 4 attention projections (H-weighted, lr={LR_ATTN})...")
    attn_results = []
    sum_init_attn = 0.0
    sum_final_attn = 0.0

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        W_key = f"model.layers.{LAYER}.self_attn.{proj}.weight"
        W_ref = load_W(idx_json, W_key)
        H = torch.load(f"{HESS_DIR}/attn_{proj}.pt", weights_only=True)["H"].numpy()
        payload = load_quantized(f"{QUANT_DIR}/attn_{proj}.pt")

        result = ft_one_linear_hweighted(
            payload, W_ref, H, lut_init,
            n_steps=N_STEPS, lr=LR_ATTN, damp=0.01, device=DEVICE,
        )

        sum_init_attn  += result["loss_init"]
        sum_final_attn += result["loss_final"]

        attn_results.append({
            "proj": proj,
            "loss_init": result["loss_init"],
            "loss_final": result["loss_final"],
            "improvement": result["improvement"],
        })
        layer_luts[f"attn_{proj}"] = result["lut_final"]

        print(f"  {proj}: {result['loss_init']:.3e} -> {result['loss_final']:.3e}  "
              f"({result['improvement']*100:.1f}%)")

    # ===== Aggregate stats =====
    print("\n" + "=" * 60)
    print("L08 AGGREGATE STATS")
    print("=" * 60)

    expert_improvements = [r["improvement"] for r in expert_results]
    expert_agg = (sum_init - sum_final) / sum_init if sum_init > 0 else 0.0
    print(f"\nExperts (n={len(expert_results)}):")
    print(f"  per-expert improvement: min={min(expert_improvements)*100:.1f}%  "
          f"max={max(expert_improvements)*100:.1f}%  "
          f"mean={np.mean(expert_improvements)*100:.1f}%")
    print(f"  TOKEN-WEIGHTED aggregate: {expert_agg*100:.2f}%  "
          f"({sum_init:.4e} -> {sum_final:.4e})")

    attn_improvements = [r["improvement"] for r in attn_results]
    attn_agg = (sum_init_attn - sum_final_attn) / sum_init_attn if sum_init_attn > 0 else 0.0
    print(f"\nAttention (n=4):")
    print(f"  per-proj improvement: {[f'{i*100:.1f}%' for i in attn_improvements]}")
    print(f"  AGGREGATE: {attn_agg*100:.2f}%  "
          f"({sum_init_attn:.4e} -> {sum_final_attn:.4e})")

    # Save tuned LUTs and stats
    torch.save(layer_luts, os.path.join(QUANT_FT_DIR, "luts.pt"))
    with open(os.path.join(QUANT_FT_DIR, "ft_stats.json"), "w") as f:
        json.dump({
            "layer": LAYER,
            "expert_results": expert_results,
            "attn_results": attn_results,
            "expert_aggregate_improvement": expert_agg,
            "attn_aggregate_improvement": attn_agg,
            "lr_expert": LR_EXPERT,
            "lr_attn": LR_ATTN,
            "n_steps": N_STEPS,
        }, f, indent=2)

    print(f"\nSaved {len(layer_luts)} tuned LUTs to {QUANT_FT_DIR}/luts.pt")

    # ===== VERDICT =====
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if expert_agg > 0.10:
        print(f"[STRONG] expert FT reduces forward MSE by {expert_agg*100:.1f}% — proceed to partial PPL")
    elif expert_agg > 0.05:
        print(f"[WEAK] expert FT reduces MSE by {expert_agg*100:.1f}% — proceed to partial PPL but expect small PPL gain")
    else:
        print(f"[STRUCTURAL] expert FT only reduces MSE by {expert_agg*100:.1f}% — pivot to no-FT story")


if __name__ == "__main__":
    main()