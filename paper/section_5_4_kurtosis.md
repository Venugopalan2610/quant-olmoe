### 5.4 Why Trellis Quantization Works: Post-RHT Weights are Essentially Gaussian

Trellis-coded quantization with the QTIP HYB codebook is designed for inputs that are approximately i.i.d. Gaussian. The quality of TCQ-based quantization depends on how close the actual weight distribution is to a Gaussian source: the closer the match, the closer the achievable distortion is to the rate-distortion lower bound for that bitrate. In this section we measure how close OLMoE's expert weights are to Gaussian *after* the RHT preprocessing step described in Section 3.3, and show that the match is essentially exact.

**Methodology.** For each of the 3072 expert weight matrices in OLMoE-1B-7B-0125 (16 layers x 64 experts x 3 projections [gate_proj, up_proj, down_proj]), we loaded the fp32 weights, applied the same random Hadamard transform used in the quantization pipeline (Section 3.3) with deterministic per-target random sign vectors, and computed the excess kurtosis (Pearson definition, Gaussian = 3.0) of the flattened transformed weights. We also measured the pre-RHT kurtosis for comparison.

**Result.** The aggregate statistics across all 3072 expert weight matrices are summarized in Table 3.

**Table 3: Kurtosis of OLMoE-1B-7B expert weights, before and after RHT.**

| Statistic | Pre-RHT | Post-RHT |
|---|---|---|
| Mean | 3.467 | **3.004** |
| Median | 3.178 | **3.003** |
| Std across tensors | 1.67 | **0.01** |
| Range | [3.00, 44.15] | [2.99, 3.70] |

Post-RHT, the mean kurtosis across all 3072 expert weight matrices is **3.004** — essentially exactly Gaussian (3.000) to three decimal places. The standard deviation across tensors drops from 1.67 (pre-RHT) to 0.01 (post-RHT), indicating that RHT produces near-perfect distributional uniformity: every expert weight matrix in the model, regardless of layer, expert index, or projection, is mapped to a distribution numerically indistinguishable from i.i.d. Gaussian.

Per-projection breakdown shows the same pattern across all three projection types: gate_proj (mean 3.007, median 3.005), up_proj (mean 3.002, median 3.002), and down_proj (mean 3.003, median 3.003). The single outlier above 3.1 in the entire 3072-matrix population is a gate_proj matrix with post-RHT kurtosis 3.70 — still within 25% of Gaussian and far below the pre-RHT maximum of 44.15.

**Interpretation.** The pre-RHT distribution of OLMoE expert weights is mildly heavy-tailed (mean kurtosis 3.47), consistent with dense-LLM weight distributions at comparable parameter scale. RHT eliminates this heavy-tailedness almost entirely: the mean drops by 0.46 and the cross-tensor variance drops by two orders of magnitude. The resulting post-RHT distribution is within 0.01 of Gaussian on average, which is well inside the tolerance at which the QTIP HYB codebook — trained on i.i.d. Gaussian samples — achieves its designed rate-distortion performance.

This is the quantitative explanation for why our 2-bit MoE quantization works as well as it does. RHT converts the problem of "quantize arbitrary expert weights" into the problem of "quantize i.i.d. Gaussian samples," and at the latter problem, QTIP's trellis codebook is near-optimal. The per-expert Hessian contribution of our work (Section 5.1) operates on top of this already-near-optimal quantizer and provides the small but consistent additional improvement observed in the main ablation; the LUT fine-tuning negative result (Section 5.3) confirms that there is essentially no remaining error for adaptive codebook adjustment to recover, because the distribution the codebook sees is the distribution the codebook was designed for.

**Relationship to QTIP's Gaussianity assumption.** The QTIP paper [Tseng et al., 2024] motivates its use of trellis-coded quantization on the premise that the random Hadamard transform produces "approximately i.i.d. Gaussian" weights, citing the incoherence bound from QuIP# [Tseng et al., 2024] and the theoretical argument that incoherent weights match the distributional assumptions of the trellis codebook. QTIP does not report numerical kurtosis measurements, relying instead on the theoretical bound and downstream quantization quality as indirect evidence. Our measurements provide the first direct numerical confirmation of the Gaussianity claim on MoE expert weights: the mean post-RHT kurtosis of 3.004 across 3072 matrices is within 0.4% of the Gaussian reference (3.000), empirically validating the assumption underlying QTIP's codebook design on a new class of weights (routed MoE experts) that the original QTIP paper did not evaluate. Importantly, MoE routing — which one might expect to produce more structured (less Gaussian) expert weight distributions due to per-expert specialization — does not prevent the distributional collapse toward Gaussianity under RHT. Whatever structural differences the routing induces, the Hadamard transform neutralizes them at the level of marginal weight distributions.