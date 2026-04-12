"""Measure per-expert post-RHT weight kurtosis for Section 5.4.

Loads each of the 3072 expert weight matrices in OLMoE-1B-7B-0125
(16 layers x 64 experts x 3 projections), applies the same RHT used
during quantization (src.rht.transform.apply_rht), and measures the
excess kurtosis (Pearson definition, Gaussian = 3.0) of the flattened
transformed weights.

Outputs:
  results/kurtosis_per_expert.json  -- per-tensor + aggregate summary

Runs on CPU only (no GPU needed), ~5 minutes for all 3072 matrices.
Safe to run while other GPU jobs are active.
"""
import os
import json
import time
import numpy as np
from scipy import stats
from safetensors import safe_open

from src.rht.transform import apply_rht, make_sign_vector

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
NUM_LAYERS = 16
NUM_EXPERTS = 64
OUT_PATH = "results/kurtosis_per_expert.json"


def safetensors_index():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)


def load_weight(idx_json, key):
    shard = idx_json["weight_map"][key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
        return f.get_tensor(key).astype(np.float32)


def main():
    os.makedirs("results", exist_ok=True)
    idx_json = safetensors_index()

    results = []
    t0 = time.time()
    print("Measuring per-expert post-RHT kurtosis across all 3072 matrices...")
    print("(pre-RHT kurtosis reported as well for comparison)")
    print()

    for layer in range(NUM_LAYERS):
        layer_t0 = time.time()
        for e in range(NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                try:
                    W = load_weight(idx_json, key)
                except Exception as ex:
                    print(f"  SKIP L{layer:02d} expert {e:02d} {proj}: {ex}")
                    continue

                # Pre-RHT kurtosis (raw weights)
                kurt_pre = float(stats.kurtosis(W.flatten(), fisher=False))

                # Deterministic per-target seeds matching the quantization pipeline
                # Note: hash(proj) is non-deterministic across processes, but kurtosis
                # of the RHT'd weights depends only on the distributional properties
                # of the transform, not the exact sign choices. We use a fixed seed
                # scheme here for reproducibility of this analysis.
                seed_l = layer * 100000 + e * 100 + {"gate_proj": 11, "up_proj": 22, "down_proj": 33}[proj]
                seed_r = seed_l + 1
                sign_l = make_sign_vector(W.shape[0], seed=seed_l)
                sign_r = make_sign_vector(W.shape[1], seed=seed_r)

                W_tilde = apply_rht(W, sign_l, sign_r)

                kurt_post = float(stats.kurtosis(W_tilde.flatten(), fisher=False))

                results.append({
                    "layer": layer,
                    "expert": e,
                    "proj": proj,
                    "shape": list(W.shape),
                    "kurt_pre_rht": kurt_pre,
                    "kurt_post_rht": kurt_post,
                })

        dt = time.time() - layer_t0
        n_done = len(results)
        print(f"  L{layer:02d}: {n_done} tensors processed ({dt:.1f}s, elapsed {time.time()-t0:.0f}s)")

    kurt_pre_all = np.array([r["kurt_pre_rht"] for r in results])
    kurt_post_all = np.array([r["kurt_post_rht"] for r in results])

    summary = {
        "num_tensors": len(results),
        "pre_rht": {
            "mean": float(kurt_pre_all.mean()),
            "median": float(np.median(kurt_pre_all)),
            "std": float(kurt_pre_all.std()),
            "min": float(kurt_pre_all.min()),
            "max": float(kurt_pre_all.max()),
        },
        "post_rht": {
            "mean": float(kurt_post_all.mean()),
            "median": float(np.median(kurt_post_all)),
            "std": float(kurt_post_all.std()),
            "min": float(kurt_post_all.min()),
            "max": float(kurt_post_all.max()),
        },
    }

    by_proj = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        subset = [r["kurt_post_rht"] for r in results if r["proj"] == proj]
        if subset:
            by_proj[proj] = {
                "count": len(subset),
                "mean": float(np.mean(subset)),
                "median": float(np.median(subset)),
                "std": float(np.std(subset)),
                "min": float(np.min(subset)),
                "max": float(np.max(subset)),
            }
    summary["post_rht_by_proj"] = by_proj

    print()
    print("=" * 70)
    print("KURTOSIS SUMMARY (Gaussian = 3.0)")
    print("=" * 70)
    print(f"Number of expert weight tensors: {summary['num_tensors']}")
    print()
    print(f"Pre-RHT  : mean={summary['pre_rht']['mean']:.3f}  median={summary['pre_rht']['median']:.3f}")
    print(f"           range=[{summary['pre_rht']['min']:.2f}, {summary['pre_rht']['max']:.2f}]  std={summary['pre_rht']['std']:.2f}")
    print()
    print(f"Post-RHT : mean={summary['post_rht']['mean']:.3f}  median={summary['post_rht']['median']:.3f}")
    print(f"           range=[{summary['post_rht']['min']:.2f}, {summary['post_rht']['max']:.2f}]  std={summary['post_rht']['std']:.2f}")
    print()
    print("Post-RHT by projection:")
    for proj, s in by_proj.items():
        print(f"  {proj:12s}: mean={s['mean']:.3f}  median={s['median']:.3f}  range=[{s['min']:.2f}, {s['max']:.2f}]")

    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "per_tensor": results}, f, indent=2)
    print()
    print(f"Saved to {OUT_PATH}")
    print(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()