"""Tripwire A.7.3.C.0: CUDA toolchain + cpp_extension JIT works."""
import sys
import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_one_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}

torch::Tensor add_one(torch::Tensor in) {
    auto out = torch::empty_like(in);
    int n = in.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}
"""


def main():
    print("=" * 60)
    print("Tripwire A.7.3.C.0: CUDA cpp_extension JIT")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available to torch")
        sys.exit(1)
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    print("\n  JIT-compiling trivial CUDA extension (first run takes ~30s)...")
    try:
        ext = load_inline(
            name="qtip_olmoe_cuda_test",
            cpp_sources=["torch::Tensor add_one(torch::Tensor in);"],
            cuda_sources=[CUDA_SOURCE],
            functions=["add_one"],
            verbose=False,
        )
    except Exception as e:
        print(f"FAIL: load_inline raised:\n{e}")
        sys.exit(1)
    print("  compiled and loaded OK")

    x = torch.arange(10, dtype=torch.float32, device="cuda")
    y = ext.add_one(x)
    expected = x + 1.0
    ok = torch.equal(y, expected)
    print(f"\n  input:    {x.cpu().tolist()}")
    print(f"  output:   {y.cpu().tolist()}")
    print(f"  expected: {expected.cpu().tolist()}")
    print(f"  [{'PASS' if ok else 'FAIL'}] kernel produces correct output")

    if ok:
        print("\nA.7.3.C.0 GATE: PASS — CUDA toolchain ready.")
        sys.exit(0)
    else:
        print("\nA.7.3.C.0 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()