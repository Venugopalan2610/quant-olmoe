"""A.6 — Calibration diagnostics.

Inspection pass over all collected Hessians. No new computation, just plots
and stats that let us verify the Hessians look sane before running BlockLDLQ.

Outputs:
  plots/calibration_diagnostics/
    eigenvalue_spectra.png       — per-layer eigenvalue distributions
    hessian_frobenius_heatmap.png — 16 x 132 energy heatmap
    rht_effect.png                — before/after RHT on 3 experts
    attention_traces.png          — 16 x 4 attention Hessian trace curves
    condition_numbers.png         — log10(cond) distribution per layer
  results/calibration_diagnostics.json — numerical summary
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.rht.transform import apply_rht, make_sign_vector

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HESSIAN_DIR = "cache/hessians"
PLOTS_DIR = "plots/calibration_diagnostics"
RESULTS_DIR = "results"

NUM_LAYERS = 16
NUM_EXPERTS = 64


def load_h(layer, kind, **kwargs):
    """Load a Hessian file and return (H, n_tokens)."""
    if kind == "attention":
        path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"attn_{kwargs['proj']}.pt")
    elif kind == "expert":
        e = kwargs["expert"]
        p = kwargs["proj"]  # 'gate_up' or 'down'
        path = os.path.join(HESSIAN_DIR, f"L{layer:02d}", f"expert_{e:02d}_{p}.pt")
    else:
        raise ValueError(kind)
    data = torch.load(path, weights_only=True)
    return data["H"], data["n_tokens"]


def diagnostic_1_eigenvalue_spectra():
    """Eigenvalue spectra for 3 representative layers.

    For each layer, show the spectrum of 4 Hessians:
      - attention q_proj
      - attention o_proj
      - expert 0 gate_up
      - expert 0 down
    """
    print("  [1/5] Eigenvalue spectra...")
    target_layers = [0, 8, 15]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False)

    for row, L in enumerate(target_layers):
        for col, (kind, spec) in enumerate([
            ("attention", {"proj": "q_proj"}),
            ("attention", {"proj": "o_proj"}),
            ("expert", {"expert": 0, "proj": "gate_up"}),
            ("expert", {"expert": 0, "proj": "down"}),
        ]):
            H, n_tok = load_h(L, kind, **spec)
            eigs = torch.linalg.eigvalsh(H).numpy()
            # Clip to positive for log scale
            eigs = np.clip(eigs, 1e-12, None)
            axes[row, col].semilogy(np.arange(len(eigs)), eigs[::-1], lw=0.8)
            if kind == "attention":
                title = f"L{L:02d} attn.{spec['proj']}"
            else:
                title = f"L{L:02d} E{spec['expert']} {spec['proj']}"
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel("eigenvalue index")
            if col == 0:
                axes[row, col].set_ylabel(f"eigenvalue (log)")

    plt.suptitle("Eigenvalue spectra (descending, log scale)", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eigenvalue_spectra.png"), dpi=100)
    plt.close()


def diagnostic_2_frobenius_heatmap():
    """16 x 132 heatmap of Hessian Frobenius norms.

    Column layout per layer:
      0-3: attn q/k/v/o
      4-67: expert gate_up for experts 0-63
      68-131: expert down for experts 0-63
    """
    print("  [2/5] Frobenius heatmap...")
    n_cols = 4 + 64 + 64  # 132
    heatmap = np.zeros((NUM_LAYERS, n_cols), dtype=np.float32)

    col_labels = []
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        col_labels.append(f"attn.{proj}")
    for e in range(NUM_EXPERTS):
        col_labels.append(f"E{e:02d}.gate_up")
    for e in range(NUM_EXPERTS):
        col_labels.append(f"E{e:02d}.down")

    for L in range(NUM_LAYERS):
        ci = 0
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            H, _ = load_h(L, "attention", proj=proj)
            heatmap[L, ci] = float(H.norm())
            ci += 1
        for e in range(NUM_EXPERTS):
            H, _ = load_h(L, "expert", expert=e, proj="gate_up")
            heatmap[L, ci] = float(H.norm())
            ci += 1
        for e in range(NUM_EXPERTS):
            H, _ = load_h(L, "expert", expert=e, proj="down")
            heatmap[L, ci] = float(H.norm())
            ci += 1

    # Log scale for visibility
    with np.errstate(divide="ignore"):
        log_heatmap = np.log10(np.clip(heatmap, 1e-6, None))

    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(log_heatmap, aspect="auto", cmap="viridis")
    ax.set_xlabel("sublayer index (0-3: attn, 4-67: expert gate_up, 68-131: expert down)")
    ax.set_ylabel("layer")
    ax.set_title("log10(Hessian Frobenius norm) across all sublayers")
    # Divider lines
    for x in [4, 68]:
        ax.axvline(x - 0.5, color="red", lw=1, alpha=0.5)
    plt.colorbar(im, ax=ax, label="log10(||H||_F)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hessian_frobenius_heatmap.png"), dpi=100)
    plt.close()

    return heatmap

def diagnostic_3_rht_effect():
    """Show before/after RHT on 3 expert weights drawn from 3 different layers."""
    print("  [3/5] RHT effect on real expert weights...")
    import json as _json
    from safetensors import safe_open

    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        idx = _json.load(f)

    target_layers = [0, 8, 15]
    expert_idx = 0

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    any_plotted = False

    for row, L in enumerate(target_layers):
        key = f"model.layers.{L}.mlp.experts.{expert_idx}.gate_proj.weight"
        if key not in idx["weight_map"]:
            print(f"    WARN: key {key!r} not in index")
            continue

        shard = idx["weight_map"][key]
        with safe_open(os.path.join(MODEL_DIR, shard), framework="numpy") as f:
            W = f.get_tensor(key).astype(np.float32)
        print(f"    L{L} E{expert_idx}: shape={W.shape}")

        sign_l = make_sign_vector(W.shape[0], seed=L * 100 + expert_idx)
        sign_r = make_sign_vector(W.shape[1], seed=L * 100 + expert_idx + 1)
        W_tilde = apply_rht(W, sign_l, sign_r)

        from scipy import stats
        kurt_b = float(stats.kurtosis(W.flatten(), fisher=False))
        kurt_a = float(stats.kurtosis(W_tilde.flatten(), fisher=False))
        std_b = float(W.std())
        std_a = float(W_tilde.std())
        print(f"      before: kurt={kurt_b:.2f} std={std_b:.5f}")
        print(f"      after:  kurt={kurt_a:.2f} std={std_a:.5f}")

        axes[row, 0].hist(W.flatten(), bins=150, density=True, alpha=0.7)
        axes[row, 0].set_title(f"L{L} E{expert_idx} gate before RHT\n"
                               f"kurt={kurt_b:.2f} std={std_b:.5f}", fontsize=10)
        axes[row, 0].set_xlabel("weight")

        x_r = np.linspace(float(W_tilde.min()), float(W_tilde.max()), 200)
        gauss = (1 / (std_a * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_r / std_a) ** 2)
        axes[row, 1].hist(W_tilde.flatten(), bins=150, density=True, alpha=0.7, label="W̃")
        axes[row, 1].plot(x_r, gauss, "r-", lw=1, label="N(0, σ²)")
        axes[row, 1].set_title(f"L{L} E{expert_idx} gate after RHT\n"
                               f"kurt={kurt_a:.2f} std={std_a:.5f}", fontsize=10)
        axes[row, 1].set_xlabel("weight")
        axes[row, 1].legend(fontsize=8)
        any_plotted = True

    plt.suptitle("RHT Gaussianification on expert weights (gate projection)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rht_effect.png"), dpi=100)
    plt.close()

    if not any_plotted:
        print("    WARN: no RHT plots were drawn")

def diagnostic_4_attention_traces():
    """Per-layer traces for the 4 attention projections."""
    print("  [4/5] Attention Hessian traces per layer...")
    traces = np.zeros((NUM_LAYERS, 4), dtype=np.float32)
    proj_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    for L in range(NUM_LAYERS):
        for i, p in enumerate(proj_names):
            H, _ = load_h(L, "attention", proj=p)
            traces[L, i] = float(H.trace())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(proj_names):
        ax.plot(range(NUM_LAYERS), traces[:, i], marker="o", label=p)
    ax.set_xlabel("layer")
    ax.set_ylabel("trace(H)")
    ax.set_yscale("log")
    ax.set_title("Attention Hessian traces per layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "attention_traces.png"), dpi=100)
    plt.close()

    return traces.tolist()


def diagnostic_5_condition_numbers():
    """Distribution of log10(condition number) per layer, computed on the
    full 2048x2048 Hessians for expert gate_up (most expensive but most
    informative for quantization conditioning)."""
    print("  [5/5] Condition number distribution...")
    cond_by_layer = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float32)

    for L in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            H, _ = load_h(L, "expert", expert=e, proj="gate_up")
            # Use eigvalsh on the full matrix
            eigs = torch.linalg.eigvalsh(H).numpy()
            eigs = np.clip(eigs, 1e-12, None)
            cond = eigs[-1] / eigs[0]
            cond_by_layer[L, e] = np.log10(cond)
        print(f"    L{L:02d}: cond range {10**cond_by_layer[L].min():.1e} .. "
              f"{10**cond_by_layer[L].max():.1e}")

    fig, ax = plt.subplots(figsize=(12, 5))
    parts = ax.violinplot(
        [cond_by_layer[L] for L in range(NUM_LAYERS)],
        positions=list(range(NUM_LAYERS)),
        showmeans=True,
        showextrema=True,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("log10(condition number)")
    ax.set_title("Expert gate_up Hessian condition number distribution\n"
                 "(each violin = 64 experts in that layer)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "condition_numbers.png"), dpi=100)
    plt.close()

    return cond_by_layer


def full_nan_sweep():
    """Walk every Hessian file, check no NaN/Inf. Returns (ok, n_files_checked)."""
    print("  [nan sweep] scanning all 2112 Hessian files...")
    n_checked = 0
    bad = []
    for L in range(NUM_LAYERS):
        d = os.path.join(HESSIAN_DIR, f"L{L:02d}")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".pt"):
                continue
            H = torch.load(os.path.join(d, f), weights_only=True)["H"]
            if torch.isnan(H).any() or torch.isinf(H).any():
                bad.append((L, f))
            n_checked += 1
    return bad, n_checked


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("A.6 — Calibration diagnostics")
    print("=" * 60)

    diagnostic_1_eigenvalue_spectra()
    fro_heatmap = diagnostic_2_frobenius_heatmap()
    diagnostic_3_rht_effect()
    attn_traces = diagnostic_4_attention_traces()
    cond_by_layer = diagnostic_5_condition_numbers()
    bad, n_checked = full_nan_sweep()

    summary = {
        "nan_sweep": {
            "files_checked": n_checked,
            "files_with_nan_or_inf": [(L, f) for L, f in bad],
            "clean": len(bad) == 0,
        },
        "attention_traces_per_layer": {
            f"L{L:02d}": {p: float(attn_traces[L][i])
                          for i, p in enumerate(("q_proj", "k_proj", "v_proj", "o_proj"))}
            for L in range(NUM_LAYERS)
        },
        "expert_frob_range_per_layer": {
            f"L{L:02d}": {
                "min_gate_up": float(fro_heatmap[L, 4:68].min()),
                "max_gate_up": float(fro_heatmap[L, 4:68].max()),
                "min_down": float(fro_heatmap[L, 68:].min()),
                "max_down": float(fro_heatmap[L, 68:].max()),
            }
            for L in range(NUM_LAYERS)
        },
        "expert_log_cond_per_layer": {
            f"L{L:02d}": {
                "median": float(np.median(cond_by_layer[L])),
                "max": float(cond_by_layer[L].max()),
                "min": float(cond_by_layer[L].min()),
            }
            for L in range(NUM_LAYERS)
        },
    }

    out_json = os.path.join(RESULTS_DIR, "calibration_diagnostics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll diagnostics written to {PLOTS_DIR}/ and {out_json}")
    print("\nHighlights:")
    print(f"  NaN sweep: {n_checked} files checked, "
          f"{len(bad)} contain NaN/Inf ({'CLEAN' if not bad else 'DIRTY'})")
    print(f"  Max log10(cond) across all expert gate_up: "
          f"{cond_by_layer.max():.2f}")
    print(f"  Median log10(cond) per layer: "
          f"L0={np.median(cond_by_layer[0]):.2f}, "
          f"L8={np.median(cond_by_layer[8]):.2f}, "
          f"L15={np.median(cond_by_layer[15]):.2f}")


if __name__ == "__main__":
    main()