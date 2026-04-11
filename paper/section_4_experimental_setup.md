## 4 Experimental Setup

### 4.1 Model

We evaluate on OLMoE-1B-7B-0125 [Muennighoff et al., 2024], a 7-billion-parameter Mixture-of-Experts language model with 1.3B active parameters per token. The architecture has 16 transformer layers, hidden dimension 2048, intermediate dimension 1024, and 64 experts per layer with top-k routing (k=8). The model is fully open-source under Apache 2.0, with weights, training data, eval code, and reference numbers all publicly released — making it the natural target for an independent researcher building on prior MoE quantization work. We use the official checkpoint from `allenai/OLMoE-1B-7B-0125` on HuggingFace, with no modifications to the base weights or architecture.

### 4.2 Calibration Data

For Hessian collection, we use 2048 sequences of 1024 tokens (totaling ~2.1M tokens) streamed from the C4 English training set [Raffel et al., 2020]. Sequences are tokenized with OLMoE's tokenizer and packed contiguously across document boundaries with EOS separators. The packed-token strategy is standard for post-training quantization: we are interested in the average input distribution to each expert, not document-level structure. The seed is fixed (`seed=0`) for reproducibility.

### 4.3 Quantization Configuration

All quantization runs use the QTIP HYB code with parameters L_bits = 16, K = 2, V = 2, Q = 9 [Tseng et al., 2024]. The codebook lookup table has shape (512, 2) and is taken directly from the official QTIP release without modification. BlockLDLQ uses block size Ty = 16 and damping `delta = 0.01 * mean(diag(H~)) * I`. RHT sign vectors are drawn from per-target seeds: layer index, expert index, and projection name are combined into a deterministic seed for each weight matrix. Quantization runs are fully reproducible given the calibration data and the seed scheme.

We compare three configurations:

1. **Per-expert H** (ours): one Hessian per expert per projection, collected from only the tokens routed to that expert. 2048 distinct Hessians for the full model.
2. **Per-layer H, unweighted mean**: one Hessian per layer per projection, computed as the unweighted mean of the 64 per-expert Hessians within that layer. 32 distinct Hessians for the full model.
3. **Per-layer H, token-weighted mean**: one Hessian per layer per projection, computed as the token-count-weighted mean of the per-expert Hessians (Section 3.2). Mathematically equivalent to recollecting Hessians without partitioning by routing — the implicit calibration of MoEQuant, MoPEQ, and EAQuant. 32 distinct Hessians for the full model.

Across all three configurations, attention Hessians are held identical (collected once during the per-expert pass) and quantization of the attention projections is unchanged. Only expert-projection Hessians vary between configurations. This isolates the routing-conditioned calibration effect from any attention-quantization noise.

### 4.4 Evaluation Protocol

**Perplexity.** We evaluate language modeling perplexity on two datasets: WikiText-2 [Merity et al., 2017] using the `wikitext-2-raw-v1` test split, and C4 [Raffel et al., 2020] using a 300,000-token sample from the validation split. Both use a sliding-window protocol with window size 2048 tokens and stride 2048 (no overlap), matching the protocol in QTIP, GPTQ, and AWQ. Within each window, we compute the next-token negative log-likelihood at every position via a single forward pass; window NLLs are summed and divided by total tokens to produce the corpus-level cross-entropy, which is then exponentiated to give perplexity. Both quantized and fp16 baselines are evaluated identically.

**Downstream tasks.** We evaluate zero-shot accuracy on five standard reasoning and knowledge benchmarks using the `lm-eval-harness` framework [Gao et al., 2024]: HellaSwag, ARC-Challenge, ARC-Easy, PIQA, and WikiText byte-perplexity. We report `acc_norm` (length-normalized accuracy) for the multiple-choice tasks and `word_perplexity` for WikiText. Batch size is auto-tuned via lm-eval-harness's `auto:4` mode, which probes for the largest batch that fits in available GPU memory; on the RTX 4080 Laptop this resolves to batch size 64 for the loglikelihood phase. We use 0-shot prompting throughout (no few-shot examples), matching the OLMoE paper's evaluation protocol.

### 4.5 Hardware and Software

All experiments run on a single NVIDIA RTX 4080 Laptop GPU (12 GB VRAM) inside Windows Subsystem for Linux 2 (WSL2) on Ubuntu 24.04.4 LTS, with 32 GB of system RAM and 64 GB of swap. The host machine is a consumer laptop. Software stack: PyTorch 2.10.0 with CUDA 12.8, NVIDIA driver 581.08, transformers 5.3.0, and lm-eval-harness from the current GitHub main branch. No multi-GPU, no distributed training, no datacenter accelerators — the entire research pipeline (Hessian collection, quantization, perplexity evaluation, and downstream task evaluation) runs within the memory and compute envelope of a single consumer laptop.

### 4.6 Reproducibility

All quantization runs are deterministic given the model checkpoint, calibration data seed, and the per-target RHT seeds described in Section 4.3. The complete pipeline — from calibration data preparation through final evaluation results — can be reproduced from the released codebase via a single shell script (`scripts/reproduce.sh`). Intermediate artifacts (per-expert Hessians, quantized bitstreams) are stored on disk and can be inspected independently. We release all bitstreams, calibration data, and evaluation results alongside the code.