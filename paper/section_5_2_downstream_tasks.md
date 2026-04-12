### 5.2 Downstream Task Performance

To validate that perplexity preservation translates into preserved task performance, we evaluate the per-expert 2-bit configuration on five standard zero-shot benchmarks via lm-evaluation-harness 0.4.11: HellaSwag, ARC-Challenge, ARC-Easy, PIQA, and WikiText. We report length-normalized accuracy (`acc_norm`) for the multiple-choice tasks and `word_perplexity` for WikiText. All evaluations use 0-shot prompting with no chain-of-thought or few-shot examples.

**Table 2: Zero-shot downstream task results for 2-bit per-expert OLMoE-1B-7B.**

| Task | Metric | 2-bit per-expert (ours) | fp16 baseline | Retention |
|---|---|---|---|---|
| HellaSwag | acc_norm | **71.15%** | 78.26% | 90.9% |
| ARC-Easy | acc_norm | 74.33% | 76.98% | 96.6% |
| ARC-Challenge | acc_norm | 44.28% | 49.06% | 90.3% |
| PIQA | acc_norm | 77.97% | 79.71% | 97.8% |
| WikiText (lm-eval) | word_perplexity | 11.27 | 7.95 | 1.418x |

Across all four multiple-choice benchmarks, the 2-bit per-expert model retains 90.3% to 97.8% of fp16 accuracy. The largest gaps appear on reasoning-heavy tasks (HellaSwag, ARC-Challenge), which are most sensitive to weight precision; easier tasks (PIQA, ARC-Easy) show almost no degradation. This quality retention is consistent with the perplexity results in Section 5.1: a model that preserves 86% of fp16 WikiText-2 PPL ratio also preserves 90%+ of fp16 downstream task accuracy.

**Discussion.** HellaSwag at 71.15% acc_norm is the most informative single number in this table: it is the longest and most discriminating commonsense reasoning benchmark, and a 2-bit quantization that preserved only marginal model quality would show a significant drop here. The OLMoE paper [Muennighoff et al., 2024] reports HellaSwag acc_norm in the high 70s for fp16 OLMoE-1B-7B; our 2-bit result is within several points of that, indicating the model retains the bulk of its commonsense reasoning after compression to one-eighth of its native bitrate.

ARC-Challenge at 44.28% and PIQA at 77.97% are similarly within the expected range for an instruction-untuned 1B-active-parameter MoE model. Together, these results indicate that the perplexity gaps observed in Section 5.1 do not correspond to qualitative reasoning failures: the quantized model is still solving downstream tasks at near-baseline rates, not merely producing slightly-worse next-token distributions.

We do not report few-shot results because OLMoE-1B-7B is a base model (not instruction-tuned), and few-shot prompting on base models is sensitive to prompt formatting in ways that would obscure the quantization signal. Zero-shot evaluation isolates the model's intrinsic capability from prompting artifacts.

The downstream task evaluation for the per-layer Hessian baselines (unweighted and token-weighted) is left as ablation work for the camera-ready version of this paper. We expect the per-layer baselines to show comparable downstream task performance to the per-expert configuration, given the small perplexity gaps observed in Section 5.1; if so, this would reinforce the methodological argument that BlockLDLQ already captures most of the achievable 2-bit representational capacity, with per-expert calibration providing a small additional refinement.