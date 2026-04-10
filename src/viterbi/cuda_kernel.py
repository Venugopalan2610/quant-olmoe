"""CUDA kernels for batched Viterbi forward DP + backtrace.

Hardcoded for L=16, k=2, V=2.
"""
import torch
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>

// Batched forward DP step.
// Grid: (256, B), block: (256,)
// Each thread owns one (batch, state) pair.
// Stores both new cum_err and the winning predecessor index (uint8) for backtrace.
__global__ void viterbi_step_batched_l16_v2(
    const float* __restrict__ cum_err_in,    // (B, 65536)
    float* __restrict__ cum_err_out,          // (B, 65536)
    uint8_t* __restrict__ backpointers_t,     // (B, 65536) slab for this timestep
    const float* __restrict__ codebook,       // (65536, 2)
    const float* __restrict__ sequences,      // (B, T)
    int t,
    int T
) {
    int b = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 65536) return;

    int seq_off = b * T;
    float w0 = sequences[seq_off + t * 2 + 0];
    float w1 = sequences[seq_off + t * 2 + 1];

    float cb0 = codebook[s * 2 + 0];
    float cb1 = codebook[s * 2 + 1];
    float d0 = cb0 - w0;
    float d1 = cb1 - w1;
    float local = d0 * d0 + d1 * d1;

    int high_bits = s >> 4;
    int batch_off = b * 65536;

    // Min over 16 predecessors, tracking best_p (lowest index on tie, matches numpy)
    float min_val = cum_err_in[batch_off + high_bits];   // p=0
    int best_p = 0;
    #pragma unroll
    for (int p = 1; p < 16; p++) {
        float v = cum_err_in[batch_off + ((p << 12) | high_bits)];
        if (v < min_val) {
            min_val = v;
            best_p = p;
        }
    }

    cum_err_out[batch_off + s] = min_val + local;
    backpointers_t[batch_off + s] = (uint8_t)best_p;
}


// Backtrace kernel: one thread per sequence walks back through backpointers.
// Reconstructs the state walk from the final argmin state.
//
// pred_state = (best_p << 12) | (state >> 4)
__global__ void viterbi_backtrace_l16_v2(
    const uint8_t* __restrict__ backpointers,  // (n_steps, B, 65536)
    const int* __restrict__ final_states,       // (B,)
    int* __restrict__ states_walk,              // (B, n_steps)
    int* __restrict__ start_states,             // (B,)
    int B,
    int n_steps
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int s = final_states[b];
    states_walk[b * n_steps + (n_steps - 1)] = s;

    for (int t = n_steps - 1; t > 0; t--) {
        // backpointers[t][b][s]
        size_t bp_idx = (size_t)t * B * 65536 + (size_t)b * 65536 + s;
        int p = (int)backpointers[bp_idx];
        s = (p << 12) | (s >> 4);
        states_walk[b * n_steps + (t - 1)] = s;
    }
    // start_state = predecessor of states_walk[0] at t=0
    size_t bp0_idx = (size_t)0 * B * 65536 + (size_t)b * 65536 + s;
    int p0 = (int)backpointers[bp0_idx];
    start_states[b] = (p0 << 12) | (s >> 4);
}


void viterbi_forward_step_batched(
    torch::Tensor cum_err_in,
    torch::Tensor cum_err_out,
    torch::Tensor backpointers_t,
    torch::Tensor codebook,
    torch::Tensor sequences,
    int64_t t,
    int64_t T,
    int64_t B
) {
    const int n_states = 65536;
    const int threads = 256;
    dim3 blocks(n_states / threads, (int)B);
    viterbi_step_batched_l16_v2<<<blocks, threads>>>(
        cum_err_in.data_ptr<float>(),
        cum_err_out.data_ptr<float>(),
        backpointers_t.data_ptr<uint8_t>(),
        codebook.data_ptr<float>(),
        sequences.data_ptr<float>(),
        (int)t, (int)T
    );
}


void viterbi_backtrace_batched(
    torch::Tensor backpointers,
    torch::Tensor final_states,
    torch::Tensor states_walk,
    torch::Tensor start_states,
    int64_t B,
    int64_t n_steps
) {
    const int threads = 128;
    int blocks = (B + threads - 1) / threads;
    viterbi_backtrace_l16_v2<<<blocks, threads>>>(
        backpointers.data_ptr<uint8_t>(),
        final_states.data_ptr<int>(),
        states_walk.data_ptr<int>(),
        start_states.data_ptr<int>(),
        (int)B, (int)n_steps
    );
}
"""

_CPP_DECL = """
void viterbi_forward_step_batched(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);
void viterbi_backtrace_batched(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);
"""


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="qtip_olmoe_viterbi_cuda_v2",
            cpp_sources=[_CPP_DECL],
            cuda_sources=[_CUDA_SRC],
            functions=[
                "viterbi_forward_step_batched",
                "viterbi_backtrace_batched",
            ],
            verbose=False,
        )
    return _ext


def viterbi_encode_v_batched_cuda(sequences, codebook):
    """Batched V=2 Viterbi encoder on CUDA, hardcoded L=16, k=2.

    Args:
        sequences: (B, T) float32 cuda tensor. T must be even.
        codebook:  (65536, 2) float32 cuda tensor.

    Returns:
        bitstreams:    (B, T/2) int32 cuda tensor (each entry is 4 bits)
        start_states:  (B,) int32 cuda tensor
        recons:        (B, T) float32 cuda tensor
        mses:          (B,) float32 cuda tensor
    """
    assert sequences.is_cuda and codebook.is_cuda
    assert sequences.dtype == torch.float32 and codebook.dtype == torch.float32
    assert codebook.shape == (65536, 2)
    B, T = sequences.shape
    assert T % 2 == 0
    n_steps = T // 2
    n_states = 65536

    sequences = sequences.contiguous()
    codebook = codebook.contiguous()

    cum_err_a = torch.zeros((B, n_states), dtype=torch.float32, device="cuda")
    cum_err_b = torch.zeros((B, n_states), dtype=torch.float32, device="cuda")
    backpointers = torch.zeros(
        (n_steps, B, n_states), dtype=torch.uint8, device="cuda",
    )

    ext = _get_ext()

    inp, out = cum_err_a, cum_err_b
    for t in range(n_steps):
        ext.viterbi_forward_step_batched(
            inp, out, backpointers[t], codebook, sequences, t, T, B,
        )
        inp, out = out, inp
    final_cum_err = inp  # (B, 65536)

    final_states = final_cum_err.argmin(dim=1).to(torch.int32)  # (B,)

    states_walk = torch.zeros((B, n_steps), dtype=torch.int32, device="cuda")
    start_states = torch.zeros((B,), dtype=torch.int32, device="cuda")
    ext.viterbi_backtrace_batched(
        backpointers, final_states, states_walk, start_states, B, n_steps,
    )

    # Bitstream: bottom kV=4 bits of each state in walk
    bitstreams = (states_walk & 0xF).to(torch.int32)

    # Reconstruction: codebook[states_walk] is (B, n_steps, 2) -> reshape to (B, T)
    recons = codebook[states_walk.long()].reshape(B, T)
    mses = ((recons - sequences) ** 2).mean(dim=1)

    return bitstreams, start_states, recons, mses