"""Tripwire: torch inverse RHT matches numpy inverse RHT (and backprops).

Verifies that the CUDA/torch implementation of inverse_rht used inside
LUT fine-tuning is bit-equivalent (to fp32 tolerance) to the numpy
reference used during quantization, and that gradients flow through it.

Run: python -m src.tripwires.test_torch_rht_numpy_rht
"""
import numpy as np
import torch
from src.rht.transform import apply_inverse_rht, make_sign_vector
from src.quantize.lut_ft import inverse_rht_torch

m, n = 2048, 1024  # OLMoE expert down_proj shape
W_tilde = np.random.randn(m, n).astype(np.float32)
sign_l = make_sign_vector(m, seed=42)
sign_r = make_sign_vector(n, seed=43)

W_np = apply_inverse_rht(W_tilde, sign_l, sign_r)

W_t = inverse_rht_torch(
    torch.from_numpy(W_tilde).cuda(),
    torch.from_numpy(sign_l).cuda(),
    torch.from_numpy(sign_r).cuda(),
).cpu().numpy()

rel_err = np.abs(W_np - W_t).max() / np.abs(W_np).max()
abs_err = np.abs(W_np - W_t).max()
print(f"max abs err: {abs_err:.2e}")
print(f"max rel err: {rel_err:.2e}")
assert rel_err < 1e-5, f"FAIL: rel_err={rel_err}"
print("[PASS] torch inverse RHT matches numpy")

# Also test with grad flowing
W_tilde_t = torch.from_numpy(W_tilde).cuda().requires_grad_(True)
sign_l_t = torch.from_numpy(sign_l).cuda()
sign_r_t = torch.from_numpy(sign_r).cuda()
W_out = inverse_rht_torch(W_tilde_t, sign_l_t, sign_r_t)
loss = W_out.pow(2).sum()
loss.backward()
print(f"grad shape: {W_tilde_t.grad.shape}")
print(f"grad finite: {torch.isfinite(W_tilde_t.grad).all().item()}")
print("[PASS] gradient flows through inverse RHT")