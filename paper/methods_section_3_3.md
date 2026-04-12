### 3.3 BlockLDLQ Trellis Quantization with RHT Preprocessing

We quantize each expert weight matrix using BlockLDLQ combined with QTIP's hybrid (HYB) bitshift codebook [Tseng et al., 2024], applied unmodified to expert weight matrices. The only change from the QTIP dense-LLM pipeline is that we substitute the per-expert Hessian H_e from Section 3.2 for the shared per-layer Hessian that QTIP uses on Llama-family dense models. The rest of this section describes the RHT preprocessing, the LDL decomposition, the trellis block structure, and the bitstream produced per target.

**Random Hadamard Transform.** Before quantization, both the weight matrix W of shape (out, in) and its corresponding Hessian H of shape (in, in) are transformed to an incoherent basis via random sign vectors and Hadamard matrices, following QuIP# and QTIP. Concretely, with random sign vectors s_L in {+1, -1}^out and s_R in {+1, -1}^in (drawn deterministically from a per-target seed) and Hadamard matrices H_out, H_in:

W~ = (1 / sqrt(out * in)) * H_out * diag(s_L) * W * diag(s_R) * H_in
H~ = diag(s_R) * H_in * H * H_in * diag(s_R) / in

The transformed weights W~ are approximately Gaussian-distributed — we measure post-RHT kurtosis averaging 3.004 across all 3072 expert weight matrices in OLMoE-1B-7B (Section 5.4), numerically within 0.4% of the Gaussian reference. This makes them well-matched to the trellis codebook, which is optimized for i.i.d. Gaussian sources. The transformed Hessian H~ encodes the same quadratic loss structure as H but in the rotated basis, so the activation-space loss (W - W_q)^T H (W - W_q) is preserved exactly under RHT. The transform is invertible: at install time, we apply the inverse transform to recover W_q in the original basis before loading it into the model (Section 3.4).

**LDL decomposition of the damped Hessian.** We add a small diagonal damping delta = 0.01 * mean(diag(H~)) * I to H~ for numerical stability, then compute a Ty-block LDL decomposition of the damped Hessian H~_damped = L D L^T, where L is unit lower triangular and D is diagonal. The LDL decomposition exposes the column-by-column error propagation structure that BlockLDLQ exploits: the off-diagonal entry L[k, j] is the linear regression coefficient of column k on column j — i.e. the amount by which column k can compensate for a quantization error in column j via the input correlation encoded in H — and D[j, j] is the residual variance of column j after conditioning on columns 0..j-1. The choice of Ty = 16 for the LDL block width matches QTIP's Llama setup and aligns with GPU MMA tile granularity.

**Block structure and BlockLDLQ loop.** Following QTIP (Section 4, Tseng et al. 2024), each quantization target is processed as a sequence of Tx x Ty = 16 x 16 = 256 weights. The Tx rows are packed along the output dimension and the Ty columns are packed along the input dimension; reshaping this 2D block into a length-256 sequence gives the trellis quantizer an effective vector dimension of 256 while keeping BlockLDLQ's column-feedback granularity at Ty = 16. This is the key QTIP trick: high trellis dimensionality (256) decouples the codebook's shaping quality from BlockLDLQ's error bound, which depends on Ty and not on TxTy.

BlockLDLQ proceeds left-to-right over the input dimension of W~ in column blocks of width Ty = 16. The algorithm maintains a residual target r, initialized as a copy of W~. At each column block j (covering columns j*Ty to (j+1)*Ty - 1):

1. The block x = r[:, j*Ty : (j+1)*Ty] is reshaped into a sequence of m/Tx sub-blocks, each of shape (Tx, Ty) = (16, 16), and each sub-block is Viterbi-decoded as a length-256 trellis sequence using the QTIP HYB codebook with vector dimension V = 2 (so the 256-length sequence walks through 128 trellis transitions, each emitting a pair of adjacent weights). The Viterbi search minimizes the squared error between the reshaped x and its trellis reconstruction x_hat.
2. The resulting quantized block W_q[:, j*Ty : (j+1)*Ty] is reshaped back to (m, Ty).
3. The quantization error e_j = r[:, j*Ty : (j+1)*Ty] - W_q[:, j*Ty : (j+1)*Ty] is propagated forward into all unquantized columns via the off-diagonal blocks of L: for each column block k > j, r[:, k*Ty : (k+1)*Ty] is updated to absorb L[k*Ty : (k+1)*Ty, j*Ty : (j+1)*Ty] * e_j.

This is coordinate descent on the LDL-decomposed quadratic loss. The total quantization loss decomposes exactly as

L(W_q) = trace((W~ - W_q)^T H~_damped (W~ - W_q)) = sum over blocks j of ||D^{1/2}[block j] * e_j||^2,

which is the sum of per-block local distortions that each Viterbi step locally minimizes. The residual target r at block j is the conditionally optimal target for block j given the quantizations already committed for blocks 0..j-1; quantizing r (rather than the original W~) is what makes the local Viterbi decisions globally optimal under the LDL decomposition.

After the loop completes, the output is a sequence of trellis bitstreams — one 16 x 16 block per sub-block of the target — that together specify W_q in the RHT basis. We discard the float reconstruction W_q at this point; it can be recomputed at install time from the bitstream alone via codebook lookup followed by inverse RHT.

**Codebook.** We use QTIP's hybrid (HYB) bitshift code with parameters L_bits = 16, k = 2, V = 2, Q = 9, Tx = Ty = 16 [Tseng et al., 2024, Section 4.2]. The codebook is a (2^Q, V) = (512, 2) lookup table trained once on i.i.d. Gaussian samples; we reuse the QTIP-released LUT without per-target adaptation and make no modifications to the code. The HYB code is a hybrid lookup-computed code that computes a pseudorandom index into the 512-entry LUT via a hash of the L_bits-bit trellis state word, then XORs a sign bit. On modern GPUs this fits in L1 cache even after duplication for bank conflicts, which is the reason QTIP's inference path is fast; see Section 3.4 for how this affects our runtime story.

**Bitstream serialization.** For each weight matrix, we save the trellis bitstreams per 16 x 16 block, the per-block start states, the RHT sign vectors s_L and s_R, and a global scale factor recovered from the RHT preprocessing. We do not save the dequantized W_q; it is reconstructed at install time. The on-disk representation occupies 3.3 GB for the full OLMoE-1B-7B model, versus 26 GB for the bf16 HuggingFace checkpoint (an 8x compression ratio on disk); see Section 4 for the breakdown.

