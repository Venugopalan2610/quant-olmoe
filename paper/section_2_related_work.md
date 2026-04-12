## 2 Related Work

### 2.1 Trellis-Coded Quantization for Dense LLMs

Post-training quantization (PTQ) has progressed rapidly from scalar quantization methods like GPTQ [Frantar et al., 2023] and AWQ [Lin et al., 2023] to vector and trellis-based methods that better exploit the structure of high-dimensional weight distributions. QuIP# [Tseng et al., 2024] introduced random Hadamard transform preprocessing as a method for converting heavy-tailed LLM weights into approximately Gaussian distributions, enabling quantization with lower distortion. QTIP [Tseng et al., 2024] extended this with trellis-coded quantization (TCQ) using the BlockLDLQ algorithm and the hybrid (HYB) bitshift codebook, achieving state-of-the-art rate-distortion performance on dense models like Llama-2-7B and Llama-2-70B at 2-bit quantization. Our work directly builds on QTIP, applying its BlockLDLQ pipeline to a new domain (MoE expert weights) with an architectural modification (per-expert Hessian collection) that we argue is essential for maintaining quality on routed-MoE architectures.

### 2.2 MoE Quantization

MoE quantization has emerged as a distinct subproblem from dense LLM quantization, with three main lines of work:

**MoEQuant** [Hu et al., 2025] addresses two specific imbalances in GPTQ-based MoE quantization: inter-expert calibration imbalance (some experts receive more calibration tokens than others) and intra-expert affinity imbalance (within an expert's routed tokens, some have higher gate weights than others). MoEQuant introduces Expert-Balanced Self-Sampling (EBSS), which generates calibration data by self-sampling from the model, optimizing for both low perplexity and balanced expert utilization, and Affinity-Guided Quantization (AGQ), which modifies the GPTQ Hessian by weighting each token's contribution by its routing gate score: `H = (X · c) X^T` rather than `X X^T`. MoEQuant operates at 4-bit and 3-bit weight quantization and reports results on Qwen-MoE-14B, DeepSeek-MoE-16B, and Mixtral-8x7B. Notably, MoEQuant does not collect *per-expert* input Hessians from routed activations: each expert in a layer still uses the layer's shared Hessian, with the affinity weighting modulating the Hessian's *contribution structure* but not its expert-conditioned input distribution. This is a meaningful methodological difference: MoEQuant captures per-token contribution variation, while we capture per-expert distribution variation.

**MoPEQ** [Chitty-Venkata et al., 2025] takes a complementary direction: rather than improving Hessian collection, it allocates *different bitwidths* to different experts based on a per-expert sensitivity metric estimated via Hutchinson trace approximation. MoPEQ tests on Vision-Language Models (DeepSeek-VL2 family and MolmoE-1B) at mixed 2/3/4 bit widths and uses SignRound (AutoRound framework) as the underlying quantization method. MoPEQ uses Hessian trace only as a sensitivity metric for bitwidth assignment; it does not collect per-expert input Hessians for use during quantization itself. Our work is orthogonal to MoPEQ's bitwidth-allocation contribution: we operate at uniform 2-bit weights and focus on improving the quality of per-expert quantization at that fixed bitwidth.

**EAQuant** [Fu et al., 2026] addresses three challenges in joint weight+activation quantization for MoE: activation outliers (via expert-aware smoothing aggregation, which combines per-expert smoothing scales into a single mergeable channel-wise vector), router quantization sensitivity (via routing consistency alignment with KL divergence), and calibration data imbalance (via expert-aware oversampling). EAQuant operates on weight+activation joint quantization regimes (W4A4, W3A4, W3A3, W2A4) using DuQuant as the baseline method. Critically for our comparison, EAQuant evaluates on OLMoE-7B, providing the only published direct comparison point for MoE quantization on the OLMoE architecture. EAQuant's calibration data balance addresses the same routing imbalance issue we observe in Section 3.2, but via a different mechanism: oversampling tokens for under-utilized experts during a shared calibration pass, rather than collecting distinct Hessians per expert. EAQuant does not collect per-expert input Hessians.

### 2.3 Differentiation

Our work occupies a previously unexplored point in the design space:

| Aspect | MoEQuant | MoPEQ | EAQuant | **Ours** |
|---|---|---|---|---|
| Per-expert input Hessians | No | No | No | **Yes** |
| Quantization method | GPTQ/AWQ | SignRound | DuQuant | **QTIP BlockLDLQ** |
| Weight bitwidth | 3-4 | 2/3/4 mixed | 2-4 | **2** |
| Activation bitwidth | 16 | 16 | 3-8 | **16** |
| Hardware tested | A6000 datacenter | Unspecified | Unspecified | **RTX 4080 Laptop, 12GB** |
| OLMoE evaluated | No | No | OLMoE-7B-0924 | **OLMoE-1B-7B-0125** |

To our knowledge, ours is the first work to (1) collect routing-conditioned per-expert input Hessians from only the tokens dispatched to each expert, (2) apply trellis-coded quantization (specifically QTIP's BlockLDLQ) to MoE expert weights, and (3) demonstrate that 2-bit weight quantization is achievable on a consumer laptop GPU for MoE models. The combination is more than the sum of its parts: BlockLDLQ's coordinate descent over column blocks naturally consumes the per-expert Hessian as a per-target input, with no algorithmic modifications needed. The contribution is methodological clarity rather than algorithmic complexity — we identify that per-expert Hessian collection is the natural calibration unit for routed-MoE architectures, and demonstrate that this single change yields measurable quality improvements (Section 5.1) while remaining fully compatible with existing trellis quantization pipelines.