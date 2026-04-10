"""Compare per-expert H vs per-layer mean H proxy losses.

Reads aggregate_stats.pt from both runs and produces the comparison
table for the paper's proxy loss section.
"""
import torch
import numpy as np

PE_PATH = "cache/quantized/aggregate_stats.pt"
PL_PATH = "cache/quantized_per_layer_H/ablation_stats.pt"


def compute_layer_means(results):
    """Results is a flat list of 3136 entries in layer order.
    Each layer has 196 targets: 4 attn + 192 expert (64 experts × 3 proj).
    """
    by_layer = {}
    for i, r in enumerate(results):
        L = i // 196
        if L not in by_layer:
            by_layer[L] = {"attn": [], "expert": []}
        if r["kind"] == "attention":
            by_layer[L]["attn"].append(r["proxy_loss"])
        else:
            by_layer[L]["expert"].append(r["proxy_loss"])
    
    result = {}
    for L in sorted(by_layer.keys()):
        result[L] = {
            "attn_mean": sum(by_layer[L]["attn"]) / len(by_layer[L]["attn"]),
            "expert_mean": sum(by_layer[L]["expert"]) / len(by_layer[L]["expert"]),
            "n_attn": len(by_layer[L]["attn"]),
            "n_expert": len(by_layer[L]["expert"]),
        }
    return result


def main():
    pe = torch.load(PE_PATH, weights_only=False)
    pl = torch.load(PL_PATH, weights_only=False)
    
    print(f"Per-expert H stats: {list(pe.keys())}")
    print(f"Per-layer H stats:  {list(pl.keys())}")
    print()
    
    # Per-expert H: aggregate the per-target results
    pe_layers = compute_layer_means(pe["results"])
    
    # Per-layer H: the script stores layer_aggregate_proxy as sums
    # Divide by 192 (expert targets per layer) for comparison
    if "layer_aggregate_proxy" in pl:
        pl_expert_means = {L: pl["layer_aggregate_proxy"][L] / 192 
                           for L in range(len(pl["layer_aggregate_proxy"]))}
    else:
        pl_layers = compute_layer_means(pl["results"])
        pl_expert_means = {L: pl_layers[L]["expert_mean"] for L in pl_layers}
    
    print(f"{'Layer':<7} {'per-expert H':>14} {'per-layer H':>14} {'ratio':>8} {'Δ%':>8}")
    print("-" * 55)
    
    ratios = []
    for L in sorted(pe_layers.keys()):
        pe_val = pe_layers[L]["expert_mean"]
        pl_val = pl_expert_means.get(L, None)
        if pl_val is None:
            continue
        ratio = pl_val / pe_val
        pct = (ratio - 1) * 100
        ratios.append(ratio)
        print(f"L{L:02d}     {pe_val:>14.4e} {pl_val:>14.4e} {ratio:>8.3f} {pct:>+7.1f}%")
    
    print("-" * 55)
    print(f"Mean ratio: {np.mean(ratios):.3f}  (per-layer H is {(np.mean(ratios)-1)*100:+.1f}% worse on average)")
    print(f"Max  ratio: {max(ratios):.3f}  (worst at layer {ratios.index(max(ratios))})")
    print(f"Min  ratio: {min(ratios):.3f}  (best at layer {ratios.index(min(ratios))})")
    
    # Save to JSON for paper figure
    import json
    comparison = {
        "per_expert_H": {f"L{L:02d}": pe_layers[L]["expert_mean"] for L in sorted(pe_layers.keys())},
        "per_layer_H":  {f"L{L:02d}": pl_expert_means[L] for L in sorted(pl_expert_means.keys())},
        "ratios":       {f"L{L:02d}": pl_expert_means[L] / pe_layers[L]["expert_mean"] 
                         for L in sorted(pe_layers.keys()) if L in pl_expert_means},
        "mean_ratio":   float(np.mean(ratios)),
    }
    with open("results/proxy_loss_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to results/proxy_loss_comparison.json")


if __name__ == "__main__":
    main()