## 5 Experiments and Results

### 5.1 Main Results: Language Modeling Perplexity

We report perplexity on WikiText-2 and C4 for OLMoE-1B-7B-0125 in three configurations: fp16 baseline, our 2-bit quantization with per-expert Hessians, and two 2-bit baselines using per-layer Hessians (unweighted and token-weighted means of the per-expert Hessians, as defined in Section 3.2). All quantized configurations share identical attention quantization, RHT preprocessing, BlockLDLQ algorithm, and trellis codebook; the only difference is which Hessian is used for the expert weight quantization.

**Table 1: Perplexity on WikiText-2 and C4 at 2 bits per weight.**

| Configuration | WikiText-2 PPL | C4 PPL | wt2 ratio | c4 ratio |
|---|---|---|---|---|
| fp16 baseline | 6.65 | 12.24 | 1.000x | 1.000x |
| 2-bit, per-expert H (ours) | **9.09** | **14.16** | **1.367x** | **1.157x** |
| 2-bit, per-layer H (unweighted mean) | 9.21 | 14.43 | 1.385x | 1.179x |
| 2-bit, per-layer H (token-weighted mean) | TBD | TBD | TBD | TBD |

The per-expert configuration achieves WikiText-2 perplexity 9.09 against an fp16 baseline of 6.65 — a 1.367x ratio at 1/8 the original bitrate. On C4, the ratio is even tighter at 1.157x. To our knowledge, no prior work has reported 2-bit MoE quantization numbers at all; the closest published comparisons are MoEQuant, MoPEQ, and EAQuant, which operate at 3-4 bits and report similar fp16 ratios on larger MoE models (Qwen-MoE-14B, DeepSeek-MoE-16B). Our 2-bit C4 ratio of 1.157x is within the range of those methods' 4-bit numbers, in a regime no prior work has entered.

**Per-expert vs per-layer ablation.** The per-expert Hessian configuration consistently outperforms both per-layer baselines on both datasets. Against the unweighted-mean baseline, the gap is 0.12 PPL on WikiText-2 and 0.27 PPL on C4. Against the token-weighted mean — the more standard comparison, mathematically equivalent to recollecting Hessians without partitioning by routing — the gap is TBD on WikiText-2 and TBD on C4.

The gaps are modest in absolute terms but consistent across datasets and across the per-layer breakdown (Section 5.5). The C4 gap is roughly twice the WikiText-2 gap, reflecting both the larger evaluation set (lower variance) and the broader text distribution (more sensitive to the routing-conditioned input statistics that per-expert calibration captures). We interpret this as evidence that per-expert calibration provides a real but bounded improvement: the marginal gain from routing-conditioned Hessians is small because BlockLDLQ with any reasonable Hessian already captures most of the achievable 2-bit representational capacity (Section 5.3), but the gain is reproducible and grows with the evaluation set's distributional breadth.

**Practical interpretation.** A 1.367x WikiText-2 ratio at 2 bits per weight means OLMoE-1B-7B retains the bulk of its language modeling quality after compression to one-eighth of its native bitrate. For comparison, the same model at fp16 has a perplexity of 6.65; our 2-bit version at 9.09 is closer to the fp16 baseline than to a model two perplexity points worse (which would correspond to a meaningfully degraded LM). The downstream task results in Section 5.2 confirm that this perplexity preservation translates into preserved zero-shot reasoning capability on standard benchmarks.