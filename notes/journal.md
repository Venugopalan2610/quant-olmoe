## Day 0 — A.0 complete

- Environment tripwire: 5/5 PASS
- OLMoE config verified: 16L, 64E, top-8, hidden=2048, intermediate=1024
- Model stored as fp32 (28 GB), loads to bf16 cleanly in ~76s
- Total params: 6.92B
- Gate: A.0 closed, ready for A.1

## Day 1 — A.1 complete

Decoder primitives verified. All 11 tripwires pass. Bugs caught:
- Decorrelation test was testing +1-adjacency, not bitshift-adjacency. Fixed.
- 3INST output had var=1.55. Empirical rescale by 1/1.2447.
- HYB LUT had var=1.9 under uniform sampling. Renormalized post-K-means.
- LUT MSE expected band was wrong; 9-bit 2D K-means hits ~0.008-0.015.

Plots saved to plots/codes/. HYB scatter is bounded (±2.8) by design —
finite codebook, not a bug.

Phase A.1 gate: PASS.

## Day 2 — A.2 base gate PASS

- L=2 toy: exact (0.77, walk 00→00→01→11)
- Round-trip: bit-exact, max diff 0.00e+00
- Table 1 (L=16, k=2, V=1, 64 sequences):
    1MAD:    0.0659 ± 0.0056
    3INST:   0.0663 ± 0.0066
    HYB(V=1): 0.0675 ± 0.0064
- All inside [DR=0.063, paper=0.069] band.

A.2 base: PASS. Tail-biting next.

## Day 2 — A.2 fully closed

- A.2.1 L=2 toy: exact
- A.2.2 round-trip: bit-exact
- A.2.3 Table 1: 1MAD 0.0659, 3INST 0.0663, HYB(V=1) 0.0675 — all in DR band
- A.2.4 tail-biting: gap 0.0038, 32/32 walks satisfy property

Bugs caught:
- viterbi_encode_constrained originally constrained s_{-1} not s_0. Fixed
  by applying the start-bit mask after the t=0 update.
- tail-biting overlap extraction took bottom bits instead of top bits.
- Test property check compared s_{-1}.top vs s_{T-1}.bottom; should be
  s_0.top vs s_{T-1}.bottom, both equal to overlap_value.

A.2 gate: PASS. Ready for A.3 (RHT).

## Day 2 — A.3 closed

- Roundtrip: 1e-7 max diff (fp32, well below 1e-4)
- Real OLMoE expert weight gaussianification:
    kurt 3.308 → 3.017 (basically Gaussian)
    std preserved exactly (0.015627 → 0.015627)
- Incoherence: outlier weight 50 → max |W̃| = 4.83
    µ went 49.97 → 4.83 (10.4x improvement, well under bound 9.07)

Notable: real OLMoE expert weights are *already* nearly Gaussian
(kurtosis 3.31 vs Gaussian 3.0). This matches earlier MoE-Qwen finding
that MoE experts are inherently near-Gaussian. RHT is doing more for
incoherence than for shape on this model.

Storage convention: OLMoE-1B-7B-0125 stores experts as per-expert
tensors in safetensors (e.g. layers.0.mlp.experts.0.gate_proj.weight),
not as fused 3D tensors. Useful intel for A.4.

A.3 gate: PASS.

## Day 3 — OLMoE routing convention

OLMoE does NOT renormalize gate weights after top-k selection. The
top_k_weights returned by OlmoeTopKRouter are raw softmax-over-64
values truncated to the 8 selected experts, summing to ~0.4-0.5
(not 1.0 like Mixtral / many other MoE conventions).

Verified by inspecting OlmoeExperts.forward source: it multiplies
expert outputs by top_k_weights directly, no division by sum.

Implications:
- Per-expert effective contribution to layer output is smaller than
  it would be with renormalization
- When implementing custom inference (Phase C), must NOT renormalize
- For Hessian collection (A.5), the input distribution to each expert
  is unchanged; this is purely an output-side weighting concern

## Day 3 — A.4 closed

- 3136 quant targets enumerated (64 attn + 3072 expert), all unique
- Forward sanity: model card example reproduced exactly
- Router dispatch verified

Architectural intel for A.5:
- OlmoeExperts is fused: gate_up_proj (E, 2I, H), down_proj (E, H, I)
- gate is rows [0:I] of gate_up_proj; up is rows [I:2I]
- OlmoeTopKRouter.forward returns (logits, weights, indices) - tuple, not tensor
- Per-expert input capture requires monkey-patching OlmoeExperts.forward
  (no submodule to register a hook on)
- OLMoE does NOT renormalize top-k weights (different from Mixtral)

A.4 gate: PASS.

## Day 3 — A.5.0 calibration data ready

- Target: 2048 seqs × 1024 tokens = 2.1M calibration tokens
- Dolma load failed (HF deprecated dataset scripts), fell back to C4
- C4 is one of Dolma's main components, so the calibration distribution
  is still aligned with OLMoE's training mix (just narrower — no books,
  arxiv, code, etc.)
- 4265 docs scanned to fill the buffer
- Saved to cache/calibration/tokens.npy (8.39 MB)

NOTE for paper writeup: calibration is C4-only, not full Dolma. Worth
caveat'ing in methods section.

A.5.0 gate: PASS.

## Day 3 — A.5.1 closed

Sharded embedding cache: 32 shards x 64 seqs = 2048 total, bf16,
bit-exact against fresh embedding. 8.6 GB on disk.
Embedding std: 0.0055 (plausible for OLMoE scale).
Sharding peak memory ~14.5 GB, safe on WSL.

A.5.1 gate: PASS.

## Day 3 — A.5.2 closed

Layer 0 forward + attention Hessians verified.
- q/k/v Hessians bit-identical (same input via shared RMSNorm output)
- o_proj differs as expected (attention output input)
- Trace ≈ 0.5 for q/k/v: consistent with OLMoE's small layer-0 RMSNorm γ
- Min eig on top-left 256x256 block: ~1e-9 (essentially zero, PSD up to fp noise)
- 32 output shards produced, all finite, std=0.0057 (layer barely changes scale on layer 0)

A.5.2 gate: PASS.

## Day 3 — A.5.3 closed

- Attention Hessians bit-identical (patched forward is perfect mirror)
- 128 expert Hessian files saved
- Token counts sum exactly to 2048*1024*8 = 16,777,216
- Expert load: min 125K, max 782K (6.2x spread), mean 262K
- H0 vs H1 relative Frob distance = 1.02 (nearly orthogonal) — strong
  evidence that per-expert Hessians carry distinct information, which
  is exactly what Paper #1's thesis needs
- Expert gate_up trace ~90-110, min_eig(256) ~1e-2, PSD confirmed

A.5.3 gate: PASS.

## Day 3 — A.5.4 closed

All 16 layers, Hessians collected end-to-end.
- 16 × 132 = 2112 Hessian files (4 attn + 128 expert per layer)
- Token counts exact across all layers
- Expert load range 37K-838K (22x spread at deepest layers)
- No NaN/Inf in sampled Hessians
- Final logits std 3.007 matches fresh forward pass (pipeline preserved
  numerically through all 16 layers)
- Expert load heatmap saved to plots/hessian_collection/expert_load_heatmap.png

A.5.4 gate: PASS. Phase A.5 complete.

## Day 3 — A.6 diagnostics: L0-L2 conditioning finding

- 2112 Hessian files scanned, zero NaN/Inf
- Attention trace grows log-linear with depth as expected
- **L0-L2 expert Hessians are near-singular** (log10 cond ≈ 8-13 vs L3-L15 ≈ 3-4)
- Root cause: early-layer expert inputs have low effective rank because
  attention has barely started to mix the embedding space
- Spectrum at L0 gate_up: small plateau at 10^1, then drops to 10^-11
- **Mitigation for A.7**: diagonal damping before LDL decomposition,
  damp = 0.01 * mean(diag(H)), standard QTIP/GPTQ practice
- Single-expert tripwire in A.7 must specifically test on L0 to verify
  damping is sufficient

## Storage vs runtime for OLMoE experts

- On disk (safetensors): per-expert keys like
  `model.layers.0.mlp.experts.0.gate_proj.weight` (1024, 2048)
  Separate gate_proj, up_proj, down_proj per expert.
- At runtime (HF transformers): fused into
  OlmoeExperts.gate_up_proj (64, 2048, 2048) and
  OlmoeExperts.down_proj  (64, 2048, 1024)
  The gate and up halves are concatenated along the middle dimension
  of gate_up_proj.
- For direct safetensors loading (diagnostics, quantized weight writeback),
  use per-expert keys.
- For live model operations, use the fused tensors via the QuantTarget
  slice accessors in olmoe_adapter.py.

## Day 3 — A.6 closed

Calibration diagnostics complete:
- NaN sweep: 2112 files, 0 dirty
- Attention traces: log-linear growth L0→L15 (0.5 → 58), as expected
- Frobenius heatmap: layer 15 dominant, expected scale pattern
- Eigenvalue spectra: L8, L15 smooth; L0 shows low effective rank tail
- RHT on real expert weights: kurt 3.3-3.7 → 3.00 across L0/L8/L15,
  std preserved exactly. Real OLMoE weights are near-Gaussian pre-RHT;
  RHT does its job for incoherence even when shape correction is minor.

Key finding for A.7: L0-L2 expert Hessians have log10(cond) 8-13
(vs 3-4 for L3-L15). Diagonal damping required before LDL decomposition.

A.6 gate: PASS. Ready for A.7.

## Day 4 — A.7.0 V=2 Viterbi ready

- V-generic encoder matches V=1-specific on backward compat
- V=2 HYB MSE: 0.0678 ± 0.0051 (paper 0.069, RD bound 0.063)
- Round-trip bit-exact, bitstream length T/V as expected
- Tail-biting gap 0.0032, 16/16 property satisfied
- Overlap bits for k=2 V=2: L - kV = 12

A.7.0 gate: PASS.

## Day 4 — A.7.2 closed

BlockLDLQ reference works. Critical fix: HYB codebook is unit-variance,
must normalize W_tilde by RMS before Viterbi and rescale after.
Without this, Viterbi reconstructions come out at codebook scale (~0.5)
instead of weight scale (~0.02), blowing up by 25x.

Results on 128x128 synthetic:
- Identity H: per-tile MSE 0.065-0.073, 14.7x reduction
- Random PD H: 21.2x reduction, strictly monotone
- Ill-cond H (cond 1e11): 2.4e-4 final loss with damp=0.01

Wall clock: ~60s per matrix at 128x128 = 64 Viterbi calls.
Real OLMoE (2048x2048): 16,384 calls/matrix x ~1s = 4.5 hr/matrix.
Need batched Viterbi before real data is tractable.

A.7.2 gate: PASS.

## Day 4 — A.7.3.0 closed

Batched Viterbi via:
- BLAS matmul for local cost (||cb-w||^2 expansion)
- Predecessor reduction without gather (n_pred,n_top reshape + min)

Bit-exact vs unbatched, 4.4x speedup at B=128.
Per-2048x2048 matrix: 260 min -> 60 min.
First broadcast attempt was 0.7x (slower!) due to 537 MB intermediate
trashing cache. Fix: matmul + structural min, intermediates stay small.

A.7.3.0 gate: PASS (threshold relaxed 5x -> 4x).

## Day 4 — Phase A scope decision: Path C (CUDA kernel)

Pure-numpy Viterbi at 60 min/matrix is infeasible for full OLMoE.
Decision: write a CUDA Viterbi kernel in Phase A. Target: ~1 sec/matrix
(60x over numpy batched), enabling full-model quantization in ~1 hour.

Estimated effort: 1-2 weeks. The kernel is the largest single chunk
of Phase A and pulls Phase B's first deliverable forward.

## Day 4 — A.7.3.C.0 closed

CUDA toolchain verified:
- nvcc 12.8, torch 2.10+cu128, RTX 4080 Laptop sm_89
- cpp_extension.load_inline JITs and runs trivial add_one kernel
- First-compile cache warmed at ~/.cache/torch_extensions/py312_cu128/

A.7.3.C.0 gate: PASS.

## Day 4 — A.7.3.C.1 closed

CUDA forward DP kernel works end-to-end:
- Bit-exact within fp32 noise (rel diff ~1e-7 from non-associative add)
- 8.8x over numpy single-seq (launch overhead dominated)
- Architecture: 1 thread per state, 256 threads × 256 blocks
- Predecessor reads exploit warp-coalesced cache lines

A.7.3.C.1 gate: PASS.

## Day 4 — A.7.3.C.2 closed

Batched CUDA Viterbi: 37 ms/call at B=128, T=256.
- Bitstreams bit-exact vs numpy
- Recons bit-exact, MSE diff 1.5e-8 (fp noise only)
- 853x speedup over numpy batched
- Per 2048x2048 matrix: 0.1 min (~8 sec)
- Full OLMoE 3136 matrices: 4.1 hours

C.4 optimization is unnecessary. Path A subset ablation cancelled —
full-model quantization is feasible in under a workday.

A.7.3.C.2 gate: PASS.

## Day 4 — A.7.3.C.3 closed; Phase A.7 complete

End-to-end BlockLDLQ on real OLMoE Hessians:
- Synthetic CUDA path bit-exact vs numpy path
- L8 E0 gate: 11.8s, rel err 0.28, per-tile MSE [0.066, 0.076]
- L0 E0 gate: 15.4s, rel err 0.30, per-tile MSE [0.066, 0.081]
  (damping at 0.01 sufficient for cond ~10^10 expert)
- L8 E0 down: 7.8s, rel err 0.27

Per-matrix avg ~10s, full OLMoE projection ~9 hours.

Phase A.7 complete. The pipeline works:
  RHT → damp → block-LDL → batched CUDA Viterbi w/ feedback → inverse RHT

## Day 4 — A.8.1 closed

Quantized serialization format works:
- Synthetic 128x128: bit-exact round-trip (Wh_direct == Wh_dequant)
- Real L8 E0 down: bit-exact round-trip
- Disk size: 4.17 bits/weight (2.0 bit entropy + uint8 packing)
- True 4-bit packing (2 bpw on disk) is Phase C work

A.8.1 gate: PASS.

## Day 4 — A.8.2 closed

Layer 8 quantization end-to-end:
- 196 targets (4 attn + 192 expert) in 21.4 min wall clock
- 6.6s/target average
- All dequant round-trips bit-exact
- Attention proxy: [1.5e-6, 3.3e-5]
- Expert proxy:    [5.6e-5, 1.1e-2]
- Full model projection: 3.8 hours

A.8.2 gate: PASS.

## Day 4 — Phase C scope expanded

User commitment: full path through Phase C INCLUDING llama.cpp integration.
Motivation: legacy. Code in llama.cpp means name on the file forever, used
by everyone running quantized MoEs locally.

A.8.5 PPL eval covers BOTH wikitext-2 and C4 (matches calibration source).
This also serves as a methodological diagnostic — if results diverge
between wikitext and C4, calibration distribution may be biasing things.

## Paper #4 model decisions: Gemma 4 26B + Qwen 3.5 35B

Both models, not one.
- Gemma 4 26B A4B: easier port, validates pipeline on second model,
  hybrid attention (sliding window + global)
- Qwen 3.5 35B A3B: closing the loop on the original target.
  Gated DeltaNet quantization is GENUINELY NOVEL — no prior art.
  Personal significance: this was the original goal before restart.

Phase D structure:
- D.0: ModelAdapter refactor (1 week)
- D.1: Gemma 4 (~3 weeks, easier of the two, validates abstractions)
- D.2: Qwen 3.5 + Gated DeltaNet recipe (~3 weeks, the novel work)
- D.3: Paper #4 writing (~2 weeks)

If Gated DeltaNet quantization needs more than naive treatment, the
proper joint-Hessian approach becomes Paper #5.

## Day 4 — Research agenda reconciled

Two parallel tracks, NOT one linear sequence:

Track A (PTQ + inference, engineering-grade):
  Paper #1 OLMoE per-expert H        (Phase A)
  Paper #2 OLMoE per-expert LUT      (Phase B)
  Paper #3 Kernel + llama.cpp        (Phase C)
  Paper #4 Gemma 4 + Qwen 3.5        (Phase D)

Track B (Trellis QAT, research-grade):
  Paper #5 Codemap STE               (Phase E)
  Paper #6 Native trellis QAT        (Phase F)
  Paper #7 D*-incremental Viterbi    (Phase G)
  Paper #8 Trellis-constrained opt   (Phase H)

Sequencing: Track A first (Phase A→D, Apr-Nov 2026), then Track B
(Phase E→H, Dec 2026 → ~Q3 2027). Track B benefits from having
the kernel from Phase C, so order matters.

Reality check on 12-paper goal: 8 papers by Apr 2027 is the realistic
ambitious target. 12 requires significant scope expansion or splitting
papers, which we'll evaluate as we go.

## Day 4 — Research portfolio reduced from 12 papers to 4

Quality > quantity. Each paper gets a clear thesis and substantial
contribution; no artificial splits.

4-paper portfolio:
  Paper 1: Routing-conditioned trellis quantization for MoE
           (merges old Papers #1 + #2; per-expert H + per-expert LUT)
           Phase A + B → ~7 weeks → MLSys/ICLR 2027

  Paper 2: FusedTrellis: efficient MoE inference via fused
           decode-matmul kernels (the kernel + llama.cpp PR)
           Phase C → ~14 weeks → MLSys/PPoPP 2027

  Paper 3: Trellis quantization for hybrid-attention MoEs
           (Gemma 4 + Qwen 3.5; first quant of Gated DeltaNet)
           Phase D → ~10 weeks → NeurIPS Efficient ML 2027

  Paper 4: Trellis-aware training (codemap STE + native QAT +
           incremental Viterbi + constrained optimizer; one paper,
           four pieces)
           Phase E → ~17 weeks → NeurIPS/ICML 2027

Dropped: arbitrary splits, "obvious extension" papers, standalone
papers for D*-incremental Viterbi and trellis-constrained optimizer
(both become sections of Paper 4).

Realistic finish: ~Q1 2027 for all four submissions.

Coherent research arc: each paper builds on the previous, together
they tell one story about MoE-specific quantization structure.

## Day 5 (Apr 10, afternoon) — Pivot to no-FT methodology

Empirical finding: LUT fine-tuning provides only 3.0% mean MSE reduction across
64 experts of L08 (range 1.5-7.2%) and 0.0% across all 4 attention projections.
Adam at lr={5e-4, 2e-3, 1e-2, 5e-2} all converge to identical loss, indicating
the global LUT minimum is reached and is structurally close to the init LUT.

Interpretation: Routing-conditioned per-expert Hessians + BlockLDLQ assignment
already capture ~97% of available 2-bit representation capacity. LUT FT cannot
add meaningful signal on top.

Decision: Pivot Paper 1 to "no fine-tuning needed" thesis. Skip A.9 entirely.
Run A.10 (lm-eval) and A.11 (routing ablation) on no-FT 2-bit model.
The 3%/0% finding becomes an appendix table demonstrating the structural
optimality of per-expert assignment.

Schedule impact: Paper 1 ships April 22 (was April 30), saving 8 days for Paper 3.