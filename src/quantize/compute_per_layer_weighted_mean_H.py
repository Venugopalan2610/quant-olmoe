"""A.11 v2 — compute TOKEN-WEIGHTED per-layer mean Hessians.

For each layer L, compute:
  H_layer_weighted = (Sum_e n_e * H_e) / (Sum_e n_e)

where H_e is the per-expert Hessian (already normalized as X_e^T X_e / n_e
during collection) and n_e is the per-expert token count.

Mathematically equivalent to (Sum_e X_e^T X_e) / (Sum_e n_e), i.e. the Hessian
you'd compute by ignoring routing and treating all calibration tokens uniformly.
"""
import os
import torch
import time

HESSIAN_DIR = "cache/hessians"
OUT_DIR = "cache/hessians_per_layer_weighted"
NUM_EXPERTS = 64
NUM_LAYERS = 16


def compute_layer_weighted_mean(layer_idx):
    layer_dir = os.path.join(HESSIAN_DIR, f"L{layer_idx:02d}")
    out_dir = os.path.join(OUT_DIR, f"L{layer_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    H_gate_up_sum = None
    H_down_sum = None
    total_n_gu = 0
    total_n_dn = 0

    for e in range(NUM_EXPERTS):
        gu = torch.load(
            os.path.join(layer_dir, f"expert_{e:02d}_gate_up.pt"),
            weights_only=True,
        )
        dn = torch.load(
            os.path.join(layer_dir, f"expert_{e:02d}_down.pt"),
            weights_only=True,
        )

        n_gu = gu["n_tokens"]
        n_dn = dn["n_tokens"]

        weighted_gu = gu["H"].double() * n_gu
        weighted_dn = dn["H"].double() * n_dn

        if H_gate_up_sum is None:
            H_gate_up_sum = weighted_gu.clone()
            H_down_sum = weighted_dn.clone()
        else:
            H_gate_up_sum += weighted_gu
            H_down_sum += weighted_dn

        total_n_gu += n_gu
        total_n_dn += n_dn

    H_gate_up = (H_gate_up_sum / total_n_gu).float()
    H_down = (H_down_sum / total_n_dn).float()

    H_gate_up = 0.5 * (H_gate_up + H_gate_up.T)
    H_down = 0.5 * (H_down + H_down.T)

    torch.save({
        "H": H_gate_up,
        "n_tokens": total_n_gu,
        "layer": layer_idx,
        "kind": "layer_weighted",
        "proj": "gate_up",
        "averaged_over_experts": NUM_EXPERTS,
    }, os.path.join(out_dir, "layer_weighted_gate_up.pt"))

    torch.save({
        "H": H_down,
        "n_tokens": total_n_dn,
        "layer": layer_idx,
        "kind": "layer_weighted",
        "proj": "down",
        "averaged_over_experts": NUM_EXPERTS,
    }, os.path.join(out_dir, "layer_weighted_down.pt"))

    print(f"  L{layer_idx:02d}: gate_up trace={float(H_gate_up.trace()):.4f}  "
          f"down trace={float(H_down.trace()):.4f}  "
          f"tokens={total_n_gu:,} (gu) / {total_n_dn:,} (dn)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Computing TOKEN-WEIGHTED per-layer mean Hessians for {NUM_LAYERS} layers...")
    t0 = time.time()
    for layer_idx in range(NUM_LAYERS):
        compute_layer_weighted_mean(layer_idx)
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Output: {OUT_DIR}/L{{NN}}/layer_weighted_{{gate_up,down}}.pt")


if __name__ == "__main__":
    main()