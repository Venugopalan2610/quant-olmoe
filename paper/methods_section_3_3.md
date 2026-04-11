### 3.3 BlockLDLQ Trellis Quantization with RHT Preprocessing

We quantize each expert weight matrix using BlockLDLQ [Tseng et al., 2024], the trellis-coded quantization algorithm introduced in QTIP. BlockLDLQ extends GPTQ-style error compensation [Frantar et al., 2023] to operate on column-block trellis codes rather than scalar quantization, while preserving the optimal column-by-column error propagation structure derived from the LDL decomposition of the input Hessian. We apply it unmodified to expert weight matrices, using the per-expert Hessian H_e collected in Section 3.2 in place of the dense layer Hessian used by QTIP.

**Random Hadamard Transform.** Before quantization, both the weight matrix W of shape (out, in) and its corresponding Hessian H of shape (in, in) are transformed to an incoherent basis via random sign vectors and Hadamard matrices, following QuIP# [Tseng et al., 2024] and QTIP. Concretely, with random sign vectors s_L in {+1, -1}^out and s_R in {+1, -1}^in (drawn deterministically from a per-target seed) and Hadamard matrices H_out, H_in:

W~ = (1 / sqrt(out * in)) * H_out * diag(s_L) * W * diag(s_R) * H_in
H~ = diag(s_R) * H_in * H * H_in * diag(s_R) / in

The transformed weights W~ are approximately Gaussian-distributed (we measure aggregate kurtosis 3.85 across OLMoE expert layers, Section 5.4), making them well-matched to the trellis codebook which is optimized for i.i.d. Gaussian sources. The transformed Hessian H~ encodes the same quadratic loss structure as H but in the rotated basis, so the activation-space loss `(W - W_q)^T H (W - W_q)` is preserved exactly. RHT is invertible: at install time, we apply the inverse transform to recover W_q in the original basis before installing it into the model.

**LDL decomposition of the damped Hessian.** We add a small diagonal damping `delta = 0.01 * mean(diag(H~))` to H~ for numerical stability, then compute the LDL decomposition:

H~_damped = L * D * L^T

where L is unit lower triangular and D is diagonal. The LDL decomposition exposes the column-by-column error propagation structure that BlockLDLQ exploits: the off-diagonal entry L[k, j] is the linear regression coefficient of column k on column j (the amount by which column k can compensate for a quantization error in column j via the input correlation encoded in H), and D[j, j] is the residual variance of column j after conditioning on columns 0..j-1.

### 3.3 BlockLDLQ Trellis Quantization with RHT Preprocessing

We quantize each expert weight matrix using BlockLDLQ [Tseng et al., 2024], the trellis-coded quantization algorithm introduced in QTIP. BlockLDLQ extends GPTQ-style error compensation [Frantar et al., 2023] to operate on column-block trellis codes rather than scalar quantization, while preserving the optimal column-by-column error propagation structure derived from the LDL decomposition of the input Hessian. We apply it unmodified to expert weight matrices, using the per-expert Hessian H_e collected in Section 3.2 in place of the dense layer Hessian used by QTIP.

**Random Hadamard Transform.** Before quantization, both the weight matrix W of shape (out, in) and its corresponding Hessian H of shape (in, in) are transformed to an incoherent basis via random sign vectors and Hadamard matrices, following QuIP# [Tseng et al., 2024] and QTIP. Concretely, with random sign vectors s_L in {+1, -1}^out and s_R in {+1, -1}^in (drawn deterministically from a per-target seed) and Hadamard matrices H_out, H_in:

W~ = (1 / sqrt(out * in)) * H_out * diag(s_L) * W * diag(s_R) * H_in
H~ = diag(s_R) * H_in * H * H_in * diag(s_R) / in

The transformed weights W~ are approximately Gaussian-distributed (we measure aggregate kurtosis 3.85 across OLMoE expert layers, Section 5.4), making them well-matched to the trellis codebook which is optimized for i.i.d. Gaussian sources. The transformed Hessian H~ encodes the same quadratic loss structure as H but in the rotated basis, so the activation-space loss `(W - W_q)^T H (W - W_q)` is preserved exactly. RHT is invertible: at install time, we apply the inverse transform to recover W_q in the original basis before installing it into the model.

**LDL decomposition of the damped Hessian.** We add a small diagonal damping `delta = 0.01 * mean(diag(H~)) * I` to H~ for numerical stability, then compute the LDL decomposition in blocks of size Ty = 16:

H~_damped = L * D * L^T

where L is unit lower triangular and D is diagonal. The LDL decomposition exposes the column-by-column error propagation structure that BlockLDLQ exploits: the off-diagonal entry L[k, j] is the linear regression coefficient of column k on column j (the amount by which column k can compensate for a quantization error in column j via the input correlation encoded in H), and D[j, j] is the residual variance of column j after conditioning on columns 0..j-1.

**The BlockLDLQ loop.** BlockLDLQ proceeds left-to-right over W~ in column blocks of size Ty = 16. Within each block, columns are jointly Viterbi-decoded into trellis codes using the QTIP HYB code with vector dimension V = 2 (so each pair of adjacent columns shares a single trellis state transition). The algorithm maintains a residual target r initialized as a copy of W~. At each column block j of width Ty:

1. The current residual target r[:, j:j+Ty] is Viterbi-decoded into trellis codes using the HYB codebook, producing a quantized reconstruction W_q[:, j:j+Ty]. The Viterbi search minimizes the local distortion `Sum_{i in block} D[i, i] * ||r[:, i] - W_q[:, i]||^2`.
2. The trellis codes are appended to the per-target bitstream.
3. The quantization error `e_j = r[:, j:j+Ty] - W_q[:, j:j+Ty]` is propagated forward into all unquantized columns via the off-diagonal block of L: for each column k beyond the current block, `r[:, k] <- r[:, k] - L[k, j:j+Ty] * e_j`.

This is coordinate descent on the LDL-decomposed loss. The total quantization loss decomposes exactly as

L(W_q) = (W~ - W_q)^T H~_damped (W~ - W_q) = Sum_j D[j, j] * ||e_j||^2

which is the sum of per-column local distortions that each Viterbi step minimizes. The residual target r[:, j] at step j is the conditionally optimal value for column j given the quantizations already committed for columns 0..j-1; quantizing r[:, j] (rather than the original W~[:, j]) is what makes the local Viterbi decisions globally optimal under the LDL decomposition.

After the loop completes, the output is a sequence of trellis codes (the bitstream), one per column block, that fully specifies W_q in the RHT basis. We discard the float reconstruction W_q at this point; it can be recomputed at install time from the bitstream alone via codebook lookup followed by inverse RHT.

**Codebook.** We use QTIP's hybrid (HYB) bitshift code with parameters L_bits = 16, k = 2, V = 2, Q = 9 [Tseng et al., 2024, Section 4.2]. This is a compute-based code (not pure lookup) that produces pseudo-random Gaussian-like reconstructions from L_bits-bit trellis state words via a cheap arithmetic transform, achieving rate-distortion performance close to the random-permutation Gaussian trellis code while admitting fast hardware decoding. The codebook lookup table has shape (512, 2) and is trained once on i.i.d. Gaussian samples; we reuse the QTIP-released LUT without per-target adaptation and make no modifications to the codebook itself.

**Bitstream serialization.** For each weight matrix, we save the trellis bitstream, the per-block start states, the RHT sign vectors s_L and s_R, and a global scale factor recovered from the RHT preprocessing. We do not save the dequantized W_q; it is reconstructed at install time. The on-disk representation occupies 3.3 GB for the full OLMoE-1B-7B model versus 13.8 GB for fp16; see Section 4 for the breakdown.