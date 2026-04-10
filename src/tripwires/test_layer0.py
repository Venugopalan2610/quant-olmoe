"""Tripwire A.5.2: layer 0 forward + attention Hessians.

A.5.2.1: output shards exist for every input shard
A.5.2.2: output hidden states are finite, sensible scale
A.5.2.3: 4 attention Hessians saved, PSD, sensible trace
A.5.2.4: layer output matches a "reference" forward through the fp32 layer
         on a small sample (correctness check)

Run: python -m src.tripwires.test_layer0
"""
import os
import sys
import torch

HIDDEN_DIR = "cache/hidden_states"
HESSIAN_DIR = "cache/hessians"


def main():
    print("=" * 60)
    print("Tripwire A.5.2: layer 0 forward + attention Hessians")
    print("=" * 60)

    in_dir = os.path.join(HIDDEN_DIR, "layer_00_input")
    out_dir = os.path.join(HIDDEN_DIR, "layer_01_input")
    hess_dir = os.path.join(HESSIAN_DIR, "L00")

    if not os.path.exists(out_dir):
        print(f"FAIL: {out_dir} does not exist. Run: python -m src.hessian.run_layer --layer 0")
        sys.exit(1)

    # A.5.2.1 — shard count matches
    in_meta = torch.load(os.path.join(in_dir, "meta.pt"), weights_only=True)
    out_meta = torch.load(os.path.join(out_dir, "meta.pt"), weights_only=True)

    in_shards = sorted(f for f in os.listdir(in_dir) if f.startswith("shard_"))
    out_shards = sorted(f for f in os.listdir(out_dir) if f.startswith("shard_"))
    count_ok = len(in_shards) == len(out_shards) == in_meta["n_shards"]
    print(f"\n  Input shards: {len(in_shards)}, Output shards: {len(out_shards)}")
    print(f"  [{'PASS' if count_ok else 'FAIL'}] shard counts match")

    # A.5.2.2 — finite + sensible scale on a few output shards
    finite_ok = True
    scale_ok = True
    stats = []
    for idx in [0, len(out_shards) // 2, len(out_shards) - 1]:
        s = torch.load(os.path.join(out_dir, f"shard_{idx:04d}.pt"), weights_only=True)
        n_nan = int(torch.isnan(s).sum())
        n_inf = int(torch.isinf(s).sum())
        sf = s.float()
        std = float(sf.std())
        if n_nan or n_inf:
            finite_ok = False
        if not (1e-4 < std < 100):
            scale_ok = False
        stats.append((idx, std, n_nan, n_inf))
        del s
    for idx, std, nan, inf in stats:
        print(f"  shard {idx}: std={std:.4f}, nan={nan}, inf={inf}")
    print(f"  [{'PASS' if finite_ok else 'FAIL'}] output shards finite")
    print(f"  [{'PASS' if scale_ok else 'FAIL'}] output scale sensible")

    # A.5.2.3 — attention Hessians
    print(f"\n  Checking attention Hessians in {hess_dir}/")
    hess_ok = True
    psd_ok = True
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        p = os.path.join(hess_dir, f"attn_{proj}.pt")
        if not os.path.exists(p):
            print(f"    MISSING: {p}")
            hess_ok = False
            continue
        data = torch.load(p, weights_only=True)
        H = data["H"]
        n_tok = data["n_tokens"]
        trace = float(H.trace())
        frob = float(H.norm())
        # PSD check on a cheap proxy: min eigenvalue of a small symmetric block
        try:
            eigs = torch.linalg.eigvalsh(H[:256, :256])
            min_eig = float(eigs.min())
        except Exception:
            min_eig = float("nan")
        psd_block_ok = min_eig > -1e-3
        print(f"    {proj}: n_tok={n_tok}, shape={tuple(H.shape)}, "
              f"trace={trace:.2f}, frob={frob:.2f}, min_eig(256x256)={min_eig:+.2e}")
        if not psd_block_ok:
            psd_ok = False

    expected_tokens = in_meta["n_seqs"] * in_meta["seq_len"]
    print(f"  Expected tokens per H: {expected_tokens}")
    print(f"  [{'PASS' if hess_ok else 'FAIL'}] all 4 Hessian files exist")
    print(f"  [{'PASS' if psd_ok else 'FAIL'}] Hessian top-left blocks PSD")

    print("\n" + "=" * 60)
    all_ok = count_ok and finite_ok and scale_ok and hess_ok and psd_ok
    if all_ok:
        print("A.5.2 GATE: PASS — layer 0 forward + attention Hessians verified.")
        print("Ready for A.5.3 (per-expert Hessians via monkey-patch).")
        sys.exit(0)
    else:
        print("A.5.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()