"""CLI driver: load a model, optionally install quantized weights, run PPL.

Usage:
    # fp16 baseline
    python -m src.eval.run_ppl --config fp16 --dataset wikitext2

    # 2-bit no-finetune
    python -m src.eval.run_ppl --config 2bit_noft --dataset wikitext2
    python -m src.eval.run_ppl --config 2bit_noft --dataset c4

    # custom output file
    python -m src.eval.run_ppl --config 2bit_noft --dataset wikitext2 \
                                --output results/ppl_2bit_noft_wt2.json
"""
import os
import sys
import json
import argparse
import time
import torch
from transformers import OlmoeForCausalLM, AutoTokenizer

from src.eval.install_quantized import install_quantized_weights
from src.eval.perplexity import (
    evaluate_perplexity,
    load_wikitext2_test,
    load_c4_validation_sample,
)

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["fp16", "2bit_noft"], required=True)
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap windows for faster debug runs")
    parser.add_argument("--c4-tokens", type=int, default=300_000,
                        help="how many C4 tokens to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: results/ppl_{config}_{dataset}.json)")
    parser.add_argument("--quant-dir", default="cache/quantized",
                        help="directory holding per-target .pt files")
    args = parser.parse_args()

    print("=" * 60)
    print(f"PPL eval: config={args.config}, dataset={args.dataset}")
    print("=" * 60)

    # Load model
    print(f"\nLoading OLMoE in bf16 on GPU...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        low_cpu_mem_usage=True)
    model.eval()
    print(f"  loaded in {time.time() - t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Optionally install quantized weights
    if args.config == "2bit_noft":
        print(f"\nInstalling 2-bit quantized weights from {args.quant_dir}...")
        install_stats = install_quantized_weights(model, quant_dir=args.quant_dir, verbose=True)
        if install_stats["n_missing"] > 0:
            print(f"FAIL: {install_stats['n_missing']} missing targets")
            sys.exit(1)
    else:
        install_stats = None

    # Load eval data
    print(f"\nLoading {args.dataset} eval data...")
    if args.dataset == "wikitext2":
        token_ids = load_wikitext2_test(tokenizer)
    else:
        token_ids = load_c4_validation_sample(tokenizer, target_tokens=args.c4_tokens)

    # Run PPL
    print(f"\nRunning PPL eval (seq_len={args.seq_len})...")
    ppl, ppl_stats = evaluate_perplexity(
        model, token_ids,
        seq_len=args.seq_len,
        device="cuda:0",
        max_windows=args.max_windows,
        verbose=True,
    )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Tag output with quant dir suffix so ablation runs don't clobber baseline
    if args.output:
        output_path = args.output
    elif args.config == "2bit_noft":
        suffix = "_per_layer_H" if "per_layer_H" in args.quant_dir else ""
        output_path = os.path.join(
            RESULTS_DIR, f"ppl_{args.config}{suffix}_{args.dataset}.json"
        )
    else:
        output_path = os.path.join(
            RESULTS_DIR, f"ppl_{args.config}_{args.dataset}.json"
        )

    result = {
        "config": args.config,
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "ppl": ppl,
        "ppl_stats": ppl_stats,
        "install_stats": install_stats,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"\nFINAL PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()