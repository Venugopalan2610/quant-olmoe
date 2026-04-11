"""Figure: per-layer proxy loss comparison across three calibration strategies.

For each of the 16 OLMoE layers, aggregates the mean proxy loss across all
expert targets (gate_proj + up_proj + down_proj, 64 experts = 192 targets per
layer) under three calibration configurations:
  - per-expert H (ours)
  - per-layer H, unweighted mean
  - per-layer H, token-weighted mean

Attention targets are excluded (they use identical Hessians across configs).
The relative gap per layer visualizes where per-expert calibration helps most.

Input:
  cache/quantized/L{00..15}/expert_*.pt
  cache/quantized_per_layer_H/L{00..15}/expert_*.pt
  cache/quantized_per_layer_weighted_H/L{00..15}/expert_*.pt

Output: figures/fig_per_layer_proxy_loss.pdf (and .png)
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

NUM_LAYERS = 16
OUT_DIR = "figures"

CONFIGS = [
    ("per-expert H (ours)", "cache/quantized", "#2d7d2d", "o", "-"),
    ("per-layer H, unweighted", "cache/quantized_per_layer_H", "#d04a4a", "s", "--"),
    ("per-layer H, token-weighted", "cache/quantized_per_layer_weighted_H", "#4a90d9", "^", "--"),
]


def aggregate_layer_proxy(quant_dir, layer):
    """Load all expert target proxy losses for one layer, return mean."""
    layer_dir = os.path.join(quant_dir, f"L{layer:02d}")
    if not os.path.isdir(layer_dir):
        return None
    proxies = []
    for fname in sorted(os.listdir(layer_dir)):
        if not fname.startswith("expert_") or not fname.endswith(".pt"):
            continue
        path = os.path.join(layer_dir, fname)
        try:
            d = torch.load(path, weights_only=False)
        except Exception:
            continue
        meta = d.get("meta", {}) if isinstance(d, dict) else {}
        pl = meta.get("proxy_loss", None)
        if pl is None and isinstance(d, dict):
            pl = d.get("proxy_loss", None)
        if pl is not None:
            proxies.append(float(pl))
    return np.mean(proxies) if proxies else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Aggregating per-layer proxy losses across configurations...")
    data = {}
    for label, quant_dir, _color, _marker, _ls in CONFIGS:
        if not os.path.isdir(quant_dir):
            print(f"  SKIP {label}: {quant_dir} does not exist")
            continue
        means = np.full(NUM_LAYERS, np.nan)
        for layer in range(NUM_LAYERS):
            m = aggregate_layer_proxy(quant_dir, layer)
            if m is not None:
                means[layer] = m
        data[label] = means
        print(f"  {label}: {np.nanmean(means):.4e} overall mean")

    layers = np.arange(NUM_LAYERS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Left panel: absolute proxy loss per layer, log scale
    ax = axes[0]
    for label, _qd, color, marker, ls in CONFIGS:
        if label not in data:
            continue
        ax.plot(layers, data[label], color=color, marker=marker, linestyle=ls,
                linewidth=1.5, markersize=5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean expert proxy loss (log scale)")
    ax.set_title("Absolute proxy loss per layer")
    ax.set_xticks(layers)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # Right panel: relative gap vs per-expert baseline
    ax = axes[1]
    if "per-expert H (ours)" in data:
        baseline = data["per-expert H (ours)"]
        for label, _qd, color, marker, ls in CONFIGS:
            if label == "per-expert H (ours)" or label not in data:
                continue
            ratio = data[label] / baseline
            ax.plot(layers, ratio, color=color, marker=marker, linestyle=ls,
                    linewidth=1.5, markersize=5, label=f"{label} / per-expert")
        ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Proxy loss ratio (baseline / per-expert)")
        ax.set_title("Per-layer gap: higher = per-expert wins more")
        ax.set_xticks(layers)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-layer expert quantization proxy loss: per-expert vs per-layer calibration",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    pdf_path = os.path.join(OUT_DIR, "fig_per_layer_proxy_loss.pdf")
    png_path = os.path.join(OUT_DIR, "fig_per_layer_proxy_loss.png")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()