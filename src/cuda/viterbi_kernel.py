"""CUDA Viterbi forward DP kernel — single-sequence reference (A.7.3.C.1).

Sub-phase C.1 scope:
  - Forward DP only (no backtrace)
  - One sequence at a time (no batching)
  - Goal: bit-exact match against numpy on final cum_err

C.2 will add backtrace and batching.

Algorithm: one CUDA thread per state s. At each timestep:
  1. Compute local_cost[s] = ||codebook[s] - w_t||^2
  2. min_pred[s] = min over the n_pred predecessors of cum_err_in
  3. cum_err_out[s] = min_pred[s] + local_cost[s]

The Python wrapper double-buffers cum_err_in / cum_err_out and launches
the kernel n_steps times. Codebook and predecessor table are uploaded
once per call (a future optimization is to cache them across calls).
"""
import torch
import numpy as np
from torch.utils.cpp_extension import load_inline

from src.viterbi.encode import precompute_codebook_v, precompute_predecessors_v


CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cfloat>

__global__ void viterbi_step_kernel(
    const float* __restrict__ cum_err_in,    // (n_states,)
    float* __restrict__ cum_err_out,          // (n_states,)
    const float* __restrict__ codebook,       // (n_states, V)
    const int* __restrict__ preds,            // (n_states, n_pred)
    const float* __restrict__ w_vec,          // (V,)
    int n_states,
    int n_pred,
    int V
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_states) return;

    // Local cost: ||codebook[s] - w_vec||^2
    float local = 0.0f;
    for (int v = 0; v < V; v++) {
        float d = codebook[s * V + v] - w_vec[v];
        local += d * d;
    }

    // Min over the n_pred predecessors of cum_err_in
    float min_pred = FLT_MAX;
    for (int p = 0; p < n_pred; p++) {
        int pred_state = preds[s * n_pred + p];
        float cost = cum_err_in[pred_state];
        if (cost < min_pred) min_pred = cost;
    }

    cum_err_out[s] = min_pred + local;
}

void viterbi_step(
    torch::Tensor cum_err_in,
    torch::Tensor cum_err_out,
    torch::Tensor codebook,
    torch::Tensor preds,
    torch::Tensor w_vec
) {
    int n_states = cum_err_in.size(0);
    int V = w_vec.size(0);
    int n_pred = preds.size(1);

    int block_size = 256;
    int n_blocks = (n_states + block_size - 1) / block_size;

    viterbi_step_kernel<<<n_blocks, block_size>>>(
        cum_err_in.data_ptr<float>(),
        cum_err_out.data_ptr<float>(),
        codebook.data_ptr<float>(),
        preds.data_ptr<int>(),
        w_vec.data_ptr<float>(),
        n_states,
        n_pred,
        V
    );
}
"""

CPP_DECLS = """
void viterbi_step(
    torch::Tensor cum_err_in,
    torch::Tensor cum_err_out,
    torch::Tensor codebook,
    torch::Tensor preds,
    torch::Tensor w_vec
);
"""


_ext = None


def get_extension():
    """JIT-compile and cache the CUDA extension."""
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="qtip_viterbi_cuda",
            cpp_sources=[CPP_DECLS],
            cuda_sources=[CUDA_SOURCE],
            functions=["viterbi_step"],
            verbose=False,
        )
    return _ext


def viterbi_forward_dp_cuda(sequence, L, k, V, decode_fn):
    """Run V-generic Viterbi forward DP on GPU.

    Args:
        sequence: 1D float32 array, length T (must be divisible by V)
        L, k, V: trellis params
        decode_fn: state->codeword function

    Returns:
        cum_err: (n_states,) numpy float32, the final accumulated errors
    """
    ext = get_extension()

    T = len(sequence)
    assert T % V == 0
    n_steps = T // V
    n_states = 1 << L

    # Build codebook and predecessor table on CPU
    codebook_np = precompute_codebook_v(L, V, decode_fn).astype(np.float32)
    preds_np = precompute_predecessors_v(L, k, V).astype(np.int32)

    device = torch.device("cuda")
    codebook = torch.from_numpy(codebook_np).to(device).contiguous()
    preds = torch.from_numpy(preds_np).to(device).contiguous()

    seq_steps_np = np.asarray(sequence, dtype=np.float32).reshape(n_steps, V)
    seq_steps = torch.from_numpy(seq_steps_np).to(device).contiguous()

    # Double-buffered cum_err
    cum_err_a = torch.zeros(n_states, dtype=torch.float32, device=device)
    cum_err_b = torch.zeros(n_states, dtype=torch.float32, device=device)

    for t in range(n_steps):
        ext.viterbi_step(cum_err_a, cum_err_b, codebook, preds, seq_steps[t])
        cum_err_a, cum_err_b = cum_err_b, cum_err_a

    torch.cuda.synchronize()
    return cum_err_a.cpu().numpy()