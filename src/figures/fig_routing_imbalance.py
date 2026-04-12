"""Figure: per-layer routing imbalance in OLMoE-1B-7B-0125 calibration.

For each of the 16 layers, plots the min/max/mean per-expert token counts across
the 64 experts, showing that the max/min ratio ranges from 2.6x (layer 0) to
17.4x (layer 15), with imbalance growing in deeper layers.

Input:  cache/hessians/L{00..15}/expert_{00..63}_gate_up.pt
Output: figures/fig_routing_imbalance.pdf (and .png)
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

HESSIAN_DIR = "cache/hessians"
OUT_DIR = "figures"
NUM_LAYERS = 16
NUM_EXPERTS = 64


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    min_counts = np.zeros(NUM_LAYERS)
    max_counts = np.zeros(NUM_LAYERS)
    mean_counts = np.zeros(NUM_LAYERS)
    ratios = np.zeros(NUM_LAYERS)

    print("Loading per-expert token counts from Hessian metadata...")
    for layer in range(NUM_LAYERS):
        counts = []
        for e in range(NUM_EXPERTS):
            path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"expert_{e:02d}_gate_up.pt")
            d = torch.load(path, weights_only=False)
            counts.append(d["n_tokens"])
        counts = np.array(counts)
        min_counts[layer] = counts.min()
        max_counts[layer] = counts.max()
        mean_counts[layer] = counts.mean()
        ratios[layer] = counts.max() / max(counts.min(), 1)
        print(f"  L{layer:02d}: min={int(counts.min()):>7,}  max={int(counts.max()):>7,}  "
              f"mean={int(counts.mean()):>7,}  max/min={ratios[layer]:.2f}")

    layers = np.arange(NUM_LAYERS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # Left panel: absolute token counts per layer (min, max, mean) on log scale
    ax = axes[0]
    ax.fill_between(layers, min_counts, max_counts, alpha=0.25, color="#4a90d9",
                     label="min-max range")
    ax.plot(layers, mean_counts, color="black", linewidth=1.5, marker="o",
            markersize=4, label="mean (uniform routing)")
    ax.plot(layers, min_counts, color="#d04a4a", linewidth=1, linestyle="--",
            marker="v", markersize=4, label="min (most-underused expert)")
    ax.plot(layers, max_counts, color="#2d7d2d", linewidth=1, linestyle="--",
            marker="^", markersize=4, label="max (most-used expert)")
    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Tokens routed to expert (log scale)")
    ax.set_title("Per-expert token count range across 16 layers")
    ax.set_xticks(layers)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.55))
    ax.grid(True, alpha=0.3, which="both")

    # Right panel: max/min ratio per layer (the imbalance metric)
    ax = axes[1]
    ax.bar(layers, ratios, color="#4a90d9", edgecolor="black", linewidth=0.5)
    ax.axhline(ratios.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean = {ratios.mean():.2f}x")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("max / min token count ratio")
    ax.set_title("Per-layer routing imbalance (max/min token ratio)")
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("OLMoE-1B-7B routing imbalance: per-expert calibration signal",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    pdf_path = os.path.join(OUT_DIR, "fig_routing_imbalance.pdf")
    png_path = os.path.join(OUT_DIR, "fig_routing_imbalance.png")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print()
    print(f"Imbalance ratio across layers: min={ratios.min():.2f}x, max={ratios.max():.2f}x, mean={ratios.mean():.2f}x")


if __name__ == "__main__":
    main()