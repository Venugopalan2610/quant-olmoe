#!/usr/bin/env bash
# reproduce.sh — Full reproduction of QTIP-MoE Paper 1 results
# Hardware: NVIDIA GPU with ≥12 GB VRAM (tested: RTX 4080 Laptop 12 GB)
# OS: Linux or WSL2 (tested: Ubuntu 22.04 under WSL2)
# Total wall time: ~12 hours end-to-end
#
# Usage:
#   bash scripts/reproduce.sh [--skip-hessians] [--skip-quant] [--skip-eval]
#
# Stages (can be resumed individually):
#   1. Environment setup        (~10 min, one-time)
#   2. Hessian collection       (~3 hr)
#   3. Quantization             (~3.5 hr, per-expert)
#   4. Ablation baselines       (~7 hr, both per-layer configs)
#   5. Perplexity evaluation    (~20 min, WikiText-2 + C4)
#   6. Downstream benchmarks    (~5 hr, lm-eval 5 tasks)
#   7. Figures                  (~2 min)

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

SKIP_HESSIANS=false
SKIP_QUANT=false
SKIP_EVAL=false

for arg in "$@"; do
  case $arg in
    --skip-hessians) SKIP_HESSIANS=true ;;
    --skip-quant)    SKIP_QUANT=true ;;
    --skip-eval)     SKIP_EVAL=true ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── STAGE 0: ENVIRONMENT ──────────────────────────────────────────────────────

log "Checking environment..."

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
python -c "
import torch
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f} GB')
assert vram >= 11.5, f'Need ≥12 GB VRAM, found {vram:.1f} GB'
"

# Pin exact versions used in paper (Section 4)
pip install -q \
  transformers==5.3.0 \
  datasets==3.2.0 \
  torch==2.10.0 \
  "lm-eval[api]==0.4.11"

log "Environment OK."

# ── STAGE 1: HESSIAN COLLECTION (~3 hr) ──────────────────────────────────────

if [ "$SKIP_HESSIANS" = false ]; then
  log "Stage 1/6: Hessian collection — estimated 3 hr on RTX 4080 Laptop..."

  log "  Step 1a: Preparing calibration data (2048 seqs × 1024 tokens from C4)"
  python -m src.hessian.prepare_calib \
    --n_seqs 2048 \
    --seq_len 1024 \
    --seed 0

  log "  Step 1b: Embedding pass (computing hidden states)"
  python -m src.hessian.embed_pass \
    --shard_size 64 \
    --batch_size 64

  log "  Step 1c: Collecting per-expert routing-conditioned Hessians (all 16 layers)"
  python -m src.hessian.collect_all \
    --start-layer 0

  log "Stage 1 complete."
else
  log "Stage 1 skipped (--skip-hessians)."
fi

# ── STAGE 2: QUANTIZATION — PER-EXPERT (~3.5 hr) ─────────────────────────────

if [ "$SKIP_QUANT" = false ]; then
  log "Stage 2/6: BlockLDLQ trellis quantization (per-expert H) — estimated 3.5 hr..."
  log "  Config: 2 bpw, L_bits=16 k=2 V=2 Tx=Ty=16, 256-length trellis sequences, damp=0.01"
  log "  Output: cache/quantized/L{00..15}/"

  python -m src.quantize.quantize_all \
    --damp 0.01

  # ── STAGE 2b: ABLATION BASELINES (~7 hr total) ────────────────────────────

  log "Stage 2b: Ablation baseline — per-layer H unweighted (~3.5 hr)..."
  python -m src.quantize.quantize_all_per_layer_H \
    --damp 0.01

  log "Stage 2c: Ablation baseline — per-layer H token-weighted (~3.5 hr)..."
  python -m src.quantize.quantize_all_per_layer_weighted_H \
    --damp 0.01

  log "Stage 2 complete."
else
  log "Stage 2 skipped (--skip-quant)."
fi

# ── STAGE 3: PERPLEXITY EVALUATION (~20 min) ─────────────────────────────────

if [ "$SKIP_EVAL" = false ]; then
  log "Stage 3/6: Perplexity evaluation — estimated 20 min..."

  mkdir -p results

  # Per-expert quantized (primary result)
  for DATASET in wikitext2 c4; do
    log "  PPL eval: per-expert / $DATASET"
    python -m src.eval.run_ppl \
      --config 2bit_noft \
      --dataset "$DATASET" \
      --quant-dir cache/quantized \
      --output "results/ppl_per_expert_${DATASET}.json"
  done

  # Per-layer H unweighted ablation
  for DATASET in wikitext2 c4; do
    log "  PPL eval: per-layer H / $DATASET"
    python -m src.eval.run_ppl \
      --config 2bit_noft \
      --dataset "$DATASET" \
      --quant-dir cache/quantized_per_layer_H \
      --output "results/ppl_per_layer_H_${DATASET}.json"
  done

  # Per-layer H token-weighted ablation
  for DATASET in wikitext2 c4; do
    log "  PPL eval: per-layer weighted H / $DATASET"
    python -m src.eval.run_ppl \
      --config 2bit_noft \
      --dataset "$DATASET" \
      --quant-dir cache/quantized_per_layer_weighted_H \
      --output "results/ppl_per_layer_weighted_H_${DATASET}.json"
  done

  # fp16 baseline
  for DATASET in wikitext2 c4; do
    log "  PPL eval: fp16 baseline / $DATASET"
    python -m src.eval.run_ppl \
      --config fp16 \
      --dataset "$DATASET" \
      --output "results/ppl_fp16_${DATASET}.json"
  done

  log "Stage 3 complete."

  # ── STAGE 4: DOWNSTREAM BENCHMARKS (~5 hr) ──────────────────────────────

  log "Stage 4/6: lm-eval downstream benchmarks — estimated 5 hr..."
  log "  Tasks: hellaswag, arc_challenge, arc_easy, piqa, wikitext"
  log "  lm-evaluation-harness 0.4.11, 0-shot, acc_norm"

  python -m src.eval.run_lm_eval \
    --config per_expert \
    --tasks wikitext,hellaswag,arc_challenge,arc_easy,piqa \
    --num-fewshot 0 \
    --output results/lmeval_per_expert.json

  # fp16 baseline
  python -m src.eval.run_lm_eval \
    --config fp16 \
    --tasks wikitext,hellaswag,arc_challenge,arc_easy,piqa \
    --num-fewshot 0 \
    --output results/lmeval_fp16.json

  log "Stage 4 complete."
fi

# ── STAGE 5: FIGURES (~2 min) ─────────────────────────────────────────────────

log "Stage 5/6: Generating figures..."
mkdir -p figures

python -m src.figures.fig_routing_imbalance
python -m src.figures.fig_kurtosis_distribution
log "Figures written to figures/"

# ── DONE ──────────────────────────────────────────────────────────────────────

log "Reproduction complete."
log ""
log "Key outputs:"
log "  PPL results:        results/ppl_*.json"
log "  Downstream results: results/lmeval_*.json"
log "  Figures:            figures/"
log ""
log "Expected numbers (Table 1, Section 5.1):"
log "  fp16          wt2=6.65   c4=12.24"
log "  per-expert    wt2=9.09   c4=14.16"
log "  per-layer-H   wt2=9.21   c4=14.43"
log "  per-layer-wH  wt2=9.18   c4=14.44"
