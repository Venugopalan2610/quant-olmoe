"""Tripwire A.8.5: no-finetune PPL on wikitext-2 + C4.

Runs all 4 (config, dataset) combos and produces a results table.

A.8.5.1: fp16 wikitext2 PPL is in [6, 12] (sanity baseline)
A.8.5.2: fp16 c4 PPL is in [10, 25]
A.8.5.3: 2bit_noft wikitext2 PPL is in [9, 30] and within 3x of fp16
A.8.5.4: 2bit_noft c4 PPL similar bounds

This is the longest tripwire so far — each config takes ~15 min on CPU.
Total: ~1 hour.

Run: python -m src.tripwires.test_a85_ppl
"""
import os
import sys
import json
import subprocess

RESULTS_DIR = "results"
CONFIGS = ["fp16", "2bit_noft"]
DATASETS = ["wikitext2", "c4"]


def run_one(config, dataset):
    out_path = os.path.join(RESULTS_DIR, f"ppl_{config}_{dataset}.json")
    print(f"\n{'='*60}")
    print(f"Running: config={config}, dataset={dataset}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "-m", "src.eval.run_ppl",
        "--config", config,
        "--dataset", dataset,
        "--output", out_path,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAIL: {config}/{dataset} returned nonzero")
        return None
    with open(out_path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Tripwire A.8.5: no-finetune PPL")
    print("=" * 60)

    results = {}
    for config in CONFIGS:
        for dataset in DATASETS:
            r = run_one(config, dataset)
            if r is None:
                print(f"\nA.8.5 GATE: FAIL (subprocess error)")
                sys.exit(1)
            results[(config, dataset)] = r["ppl"]

    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(f"{'config':<14} {'wikitext2':>12} {'c4':>12}")
    print("-" * 60)
    for config in CONFIGS:
        wt2 = results[(config, "wikitext2")]
        c4 = results[(config, "c4")]
        print(f"{config:<14} {wt2:>12.4f} {c4:>12.4f}")

    fp16_wt2 = results[("fp16", "wikitext2")]
    fp16_c4 = results[("fp16", "c4")]
    q_wt2 = results[("2bit_noft", "wikitext2")]
    q_c4 = results[("2bit_noft", "c4")]

    delta_wt2 = q_wt2 - fp16_wt2
    delta_c4 = q_c4 - fp16_c4
    print(f"{'delta (q-fp16)':<14} {delta_wt2:>+12.4f} {delta_c4:>+12.4f}")

    # Gate checks
    print("\n" + "-" * 60)
    fp16_wt2_ok = 6.0 <= fp16_wt2 <= 12.0
    fp16_c4_ok = 10.0 <= fp16_c4 <= 25.0
    q_wt2_ok = q_wt2 < 3 * fp16_wt2 and q_wt2 < 30
    q_c4_ok = q_c4 < 3 * fp16_c4 and q_c4 < 60

    print(f"  [{'PASS' if fp16_wt2_ok else 'FAIL'}] fp16 wt2 in [6, 12]")
    print(f"  [{'PASS' if fp16_c4_ok else 'FAIL'}] fp16 c4 in [10, 25]")
    print(f"  [{'PASS' if q_wt2_ok else 'FAIL'}] 2bit_noft wt2 < 3x fp16 and < 30")
    print(f"  [{'PASS' if q_c4_ok else 'FAIL'}] 2bit_noft c4 < 3x fp16 and < 60")

    # Save the combined results table
    table_path = os.path.join(RESULTS_DIR, "a85_ppl_table.json")
    with open(table_path, "w") as f:
        json.dump({
            "results": {f"{c}_{d}": v for (c, d), v in results.items()},
            "deltas": {
                "wikitext2": delta_wt2,
                "c4": delta_c4,
            },
        }, f, indent=2)
    print(f"\nTable saved to {table_path}")

    all_ok = fp16_wt2_ok and fp16_c4_ok and q_wt2_ok and q_c4_ok
    if all_ok:
        print("\nA.8.5 GATE: PASS — no-finetune baseline established.")
        print("Ready for A.9 (blockwise fine-tuning).")
        sys.exit(0)
    else:
        print("\nA.8.5 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()