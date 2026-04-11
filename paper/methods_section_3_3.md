### 3.3 BlockLDLQ Trellis Quantization with RHT Preprocessing

We quantize each expert weight matrix using BlockLDLQ [Tseng et al., 2024], the trellis-coded quantization algorithm introduced in QTIP. BlockLDLQ extends GPTQ-style error compensation [Frantar et al., 2023] to operate on column-block trellis codes rather than scalar quantization, while preserving the optimal column-by-column error propagation structure derived from the LDL decomposition of the input Hessian. We apply it unmodified to expert weight matrices, using the per-expert Hessian H_e collected in Section 3.2 in place of the dense layer Hessian used by QTIP.

**Random Hadamard Transform.** Before quantization, both the weight matrix W of shape (out, in) and its corresponding Hessian H of shape (in, in) are transformed to an incoherent basis via random sign vectors and Hadamard matrices, following QuIP# [Tseng et al., 2024] and QTIP. Concretely, with random sign vectors s_L in {+1, -1}^out and s_R in {+1, -1}^in (drawn deterministically from a per-target seed) and Hadamard matrices H_out, H_in:

W~ = (1 / sqrt(out * in)) * H_out * diag(s_L) * W * diag(s_R) * H_in
H~ = diag(s_R) * H_in * H * H_in * diag(s_R) / in

