"""Tripwire A.5.3: layer 0 with per-expert Hessian collection.

A.5.3.1: Attention Hessians bit-identical after re-running with experts
A.5.3.2: All 128 expert Hessian files exist (64 experts x {gate_up, down})
A.5.3.3: Per-expert token counts sum to n_total_tokens * top_k
A.5.3.4: Per-expert Hessians are distinct (not all identical)
A.5.3.5: PSD sanity on a sample of expert Hessians

Run: python -m src.tripwires.test_layer0_experts
"""
import os
import sys
import shutil
import subprocess
import torch

HESSIAN_DIR = "cache/hessians/L00"
BACKUP_DIR = "cache/hessians/L00_attn_bak"


def main():
    print("=" * 60)
    print("Tripwire A.5.3: layer 0 with per-expert Hessians")
    print("=" * 60)

    # Back up the attention Hessians from A.5.2
    if not os.path.exists(HESSIAN_DIR):
        print(f"FAIL: {HESSIAN_DIR} doesn't exist. Run A.5.2 first.")
        sys.exit(1)

    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    os.makedirs(BACKUP_DIR)
    for f in os.listdir(HESSIAN_DIR):
        if f.startswith("attn_"):
            shutil.copy2(os.path.join(HESSIAN_DIR, f), os.path.join(BACKUP_DIR, f))
    print(f"  Backed up {len(os.listdir(BACKUP_DIR))} attention Hessian files to {BACKUP_DIR}")

    # Run layer 0 with expert collection
    print(f"\n  Running layer 0 with --with-experts (this will take a few minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "src.hessian.run_layer", "--layer", "0", "--with-experts"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("FAIL: run_layer returned nonzero")
        sys.exit(1)

    # A.5.3.1 — attention Hessians bit-identical
    print(f"\nA.5.3.1: attention Hessian bit-identity")
    print("-" * 60)
    attn_ok = True
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        old = torch.load(os.path.join(BACKUP_DIR, f"attn_{proj}.pt"), weights_only=True)
        new = torch.load(os.path.join(HESSIAN_DIR, f"attn_{proj}.pt"), weights_only=True)
        identical = torch.equal(old["H"], new["H"]) and old["n_tokens"] == new["n_tokens"]
        if not identical:
            max_diff = float((old["H"].float() - new["H"].float()).abs().max())
            print(f"  attn_{proj}: DIFFER (max abs diff {max_diff:.2e})")
            attn_ok = False
        else:
            print(f"  attn_{proj}: bit-identical")
    print(f"  [{'PASS' if attn_ok else 'FAIL'}] attention Hessians unchanged by expert patch")

    # Clean up backup
    shutil.rmtree(BACKUP_DIR)

    # A.5.3.2 — expert files exist
    print(f"\nA.5.3.2: expert Hessian file count")
    print("-" * 60)
    expert_files = sorted(
        f for f in os.listdir(HESSIAN_DIR) if f.startswith("expert_")
    )
    expected = 64 * 2
    files_ok = len(expert_files) == expected
    print(f"  Found {len(expert_files)} expert files, expected {expected}")
    print(f"  [{'PASS' if files_ok else 'FAIL'}] all expert files present")

    # A.5.3.3 — token counts sum correctly
    print(f"\nA.5.3.3: per-expert token count totals")
    print("-" * 60)
    total_gu_tokens = 0
    total_dn_tokens = 0
    gu_counts = []
    dn_counts = []
    for e in range(64):
        gu = torch.load(os.path.join(HESSIAN_DIR, f"expert_{e:02d}_gate_up.pt"),
                        weights_only=True)
        dn = torch.load(os.path.join(HESSIAN_DIR, f"expert_{e:02d}_down.pt"),
                        weights_only=True)
        total_gu_tokens += gu["n_tokens"]
        total_dn_tokens += dn["n_tokens"]
        gu_counts.append(gu["n_tokens"])
        dn_counts.append(dn["n_tokens"])

    # Expected: 2048 seqs * 1024 tokens * top_k=8 = 16,777,216
    n_tokens_total = 2048 * 1024
    top_k = 8
    expected_dispatches = n_tokens_total * top_k

    print(f"  Total gate_up dispatches: {total_gu_tokens:,}")
    print(f"  Total down dispatches:    {total_dn_tokens:,}")
    print(f"  Expected:                 {expected_dispatches:,}")
    print(f"  Expert load range (gate_up): "
          f"min={min(gu_counts):,}, max={max(gu_counts):,}, "
          f"mean={sum(gu_counts)//64:,}")

    gu_ok = total_gu_tokens == expected_dispatches
    dn_ok = total_dn_tokens == expected_dispatches
    min_load_ok = min(gu_counts) > 10000  # at least 10K tokens per expert
    print(f"  [{'PASS' if gu_ok else 'FAIL'}] gate_up counts sum exactly")
    print(f"  [{'PASS' if dn_ok else 'FAIL'}] down counts sum exactly")
    print(f"  [{'PASS' if min_load_ok else 'FAIL'}] min expert load > 10K tokens")

    # A.5.3.4 — distinct Hessians
    print(f"\nA.5.3.4: per-expert Hessians distinct")
    print("-" * 60)
    H0 = torch.load(os.path.join(HESSIAN_DIR, "expert_00_gate_up.pt"),
                    weights_only=True)["H"]
    H1 = torch.load(os.path.join(HESSIAN_DIR, "expert_01_gate_up.pt"),
                    weights_only=True)["H"]
    H5 = torch.load(os.path.join(HESSIAN_DIR, "expert_05_gate_up.pt"),
                    weights_only=True)["H"]
    d01 = float((H0 - H1).norm() / H0.norm())
    d05 = float((H0 - H5).norm() / H0.norm())
    print(f"  Relative Frobenius distance H0 vs H1: {d01:.4f}")
    print(f"  Relative Frobenius distance H0 vs H5: {d05:.4f}")
    distinct_ok = d01 > 0.01 and d05 > 0.01
    print(f"  [{'PASS' if distinct_ok else 'FAIL'}] expert Hessians are distinct (>1% rel. diff)")

    # A.5.3.5 — PSD sanity on random sample
    print(f"\nA.5.3.5: PSD sanity on 4 random expert Hessians")
    print("-" * 60)
    import random
    random.seed(0)
    sample = random.sample(range(64), 4)
    psd_ok = True
    for e in sample:
        gu = torch.load(os.path.join(HESSIAN_DIR, f"expert_{e:02d}_gate_up.pt"),
                        weights_only=True)
        eigs = torch.linalg.eigvalsh(gu["H"][:256, :256])
        min_eig = float(eigs.min())
        trace = float(gu["H"].trace())
        print(f"  expert {e} gate_up: n_tok={gu['n_tokens']:,} "
              f"trace={trace:.4f} min_eig(256)={min_eig:+.2e}")
        if min_eig < -1e-4:
            psd_ok = False
    print(f"  [{'PASS' if psd_ok else 'FAIL'}] sampled Hessians PSD")

    print("\n" + "=" * 60)
    all_ok = attn_ok and files_ok and gu_ok and dn_ok and min_load_ok and distinct_ok and psd_ok
    if all_ok:
        print("A.5.3 GATE: PASS — per-expert Hessian collection verified.")
        print("Ready for A.5.4 (run all 16 layers).")
        sys.exit(0)
    else:
        print("A.5.3 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()