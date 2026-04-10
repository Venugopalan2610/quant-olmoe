"""A.8.3 — Full OLMoE quantization driver.

Iterates over all 16 layers, calling quantize_layer on each. Supports resume
via --start-layer if a previous run crashed mid-way.

Output: cache/quantized/L00..L15/, each with 196 .pt files
        cache/quantized/aggregate_stats.pt
"""
import os
import sys
import time
import argparse
import torch

from src.quantize.quantize_layer import quantize_layer

QUANTIZED_DIR = "cache/quantized"
NUM_LAYERS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=NUM_LAYERS)
    parser.add_argument("--damp", type=float, default=0.01)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    print(f"Quantizing layers [{args.start_layer}, {args.end_layer})")
    print(f"Damping coefficient: {args.damp}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    grand_start = time.time()
    all_results = []
    layer_times = []

    for layer_idx in range(args.start_layer, args.end_layer):
        print(f"\n=== Layer {layer_idx} ===")
        t0 = time.time()
        results = quantize_layer(
            layer_idx=layer_idx,
            damp=args.damp,
            verbose=True,
        )
        layer_time = time.time() - t0
        layer_times.append(layer_time)
        all_results.extend(results)

        # Per-layer summary
        attn = [r for r in results if r["kind"] == "attention"]
        expert = [r for r in results if r["kind"] == "expert"]
        attn_mean = sum(r["proxy_loss"] for r in attn) / max(len(attn), 1)
        expert_mean = sum(r["proxy_loss"] for r in expert) / max(len(expert), 1)
        elapsed_total = time.time() - grand_start
        remaining = args.end_layer - layer_idx - 1
        eta = (elapsed_total / (layer_idx - args.start_layer + 1)) * remaining

        print(f"  layer {layer_idx} done in {layer_time:.0f}s "
              f"({layer_time/60:.1f} min)")
        print(f"  attn proxy mean: {attn_mean:.3e}")
        print(f"  expert proxy mean: {expert_mean:.3e}")
        print(f"  total elapsed: {elapsed_total/60:.1f} min")
        print(f"  ETA remaining: {eta/60:.1f} min")

    grand_total = time.time() - grand_start

    # Save aggregate stats
    stats_path = os.path.join(QUANTIZED_DIR, "aggregate_stats.pt")
    torch.save({
        "results": all_results,
        "layer_times": layer_times,
        "total_seconds": grand_total,
        "layers_processed": list(range(args.start_layer, args.end_layer)),
        "damp": args.damp,
    }, stats_path)

    print(f"\n=== All {args.end_layer - args.start_layer} layers done ===")
    print(f"Total wall clock: {grand_total:.0f}s ({grand_total/60:.1f} min, "
          f"{grand_total/3600:.2f} hours)")
    print(f"Per-layer average: {sum(layer_times)/len(layer_times):.0f}s")
    print(f"Aggregate stats saved to: {stats_path}")


if __name__ == "__main__":
    main()