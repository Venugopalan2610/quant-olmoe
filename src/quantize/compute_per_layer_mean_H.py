"""A.11 prep — compute per-layer mean Hessians from per-expert Hessians.

For each layer L, compute:
  H_layer_gate_up = mean over experts of expert_e_gate_up_H
  H_layer_down    = mean over experts of expert_e_down_H

These represent "what if you ignored routing and used layer-average input statistics."
The Paper 1 thesis is that per-expert H beats this.
"""
import os
import torch
import time

HESSIAN_DIR = "cache/hessians"
OUT_DIR = "cache/hessians_per_layer_mean"
NUM_EXPERTS = 64
NUM_LAYERS = 16


def compute_layer_mean(layer_idx):
    """Average all 64 experts' Hessians in one layer for both gate_up and down."""
    layer_dir = os.path.join(HESSIAN_DIR, f"L{layer_idx:02d}")
    out_dir = os.path.join(OUT_DIR, f"L{layer_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # Initialize accumulators with first expert (to get shapes)
    first_gate_up = torch.load(
        os.path.join(layer_dir, "expert_00_gate_up.pt"), weights_only=True
    )
    first_down = torch.load(
        os.path.join(layer_dir, "expert_00_down.pt"), weights_only=True
    )

    H_gate_up = first_gate_up["H"].clone().double()
    H_down = first_down["H"].clone().double()
    n_tokens_gu = first_gate_up["n_tokens"]
    n_tokens_dn = first_down["n_tokens"]

    for e in range(1, NUM_EXPERTS):
        gu = torch.load(
            os.path.join(layer_dir, f"expert_{e:02d}_gate_up.pt"), weights_only=True
        )
        dn = torch.load(
            os.path.join(layer_dir, f"expert_{e:02d}_down.pt"), weights_only=True
        )
        H_gate_up += gu["H"].double()
        H_down += dn["H"].double()
        n_tokens_gu += gu["n_tokens"]
        n_tokens_dn += dn["n_tokens"]

    H_gate_up /= NUM_EXPERTS
    H_down /= NUM_EXPERTS
    H_gate_up = (0.5 * (H_gate_up + H_gate_up.T)).float()
    H_down = (0.5 * (H_down + H_down.T)).float()

    # Save with same schema as per-expert files so the loader is uniform
    torch.save({
        "H": H_gate_up,
        "n_tokens": n_tokens_gu,
        "layer": layer_idx,
        "kind": "layer_mean",
        "proj": "gate_up",
        "averaged_over_experts": NUM_EXPERTS,
    }, os.path.join(out_dir, "layer_mean_gate_up.pt"))

    torch.save({
        "H": H_down,
        "n_tokens": n_tokens_dn,
        "layer": layer_idx,
        "kind": "layer_mean",
        "proj": "down",
        "averaged_over_experts": NUM_EXPERTS,
    }, os.path.join(out_dir, "layer_mean_down.pt"))

    print(f"  L{layer_idx:02d}: gate_up trace={float(H_gate_up.trace()):.4f}  "
          f"down trace={float(H_down.trace()):.4f}  "
          f"tokens={n_tokens_gu:,} (gu) / {n_tokens_dn:,} (dn)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Computing per-layer mean Hessians for {NUM_LAYERS} layers...")
    t0 = time.time()
    for layer_idx in range(NUM_LAYERS):
        compute_layer_mean(layer_idx)
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Output: {OUT_DIR}/L{{NN}}/layer_mean_{{gate_up,down}}.pt")


if __name__ == "__main__":
    main()