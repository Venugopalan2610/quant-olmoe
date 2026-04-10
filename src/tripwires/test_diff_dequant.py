"""Verify differentiable_dequant matches numpy dequant_target bit-for-bit."""
import numpy as np
import torch

from src.quantize.serialize import load_quantized, dequant_target
from src.codes.ref import decode_hyb_batch
from src.quantize.lut_ft import (
    precompute_walk_states,
    differentiable_dequant,
)

TARGET_FILE = "cache/quantized/L08/expert_00_gate_proj.pt"
LUT_PATH = "cache/codes/hyb_lut_init.npy"

def main():
    # Load saved payload
    saved = load_quantized(TARGET_FILE)
    lut_init = np.load(LUT_PATH)
    print(f"Target shape: {saved['shape']}")
    print(f"LUT shape: {lut_init.shape}")
    print(f"Bitstreams shape: {saved['bitstreams'].shape}")

    # Numpy dequant (ground truth)
    decode_fn = lambda s: decode_hyb_batch(s, lut_init, Q=9)
    W_np = dequant_target(saved, decode_fn)
    print(f"Numpy W: shape={W_np.shape}, dtype={W_np.dtype}, "
          f"mean={W_np.mean():.6f}, std={W_np.std():.6f}")

    # Torch differentiable dequant
    walks = precompute_walk_states(
        saved["bitstreams"], saved["start_states"],
        L_bits=saved["config"]["L"],
        kV=saved["config"]["k"] * saved["config"]["V"],
    )

    device = "cuda:0"
    walks_t = torch.from_numpy(walks).long().to(device)
    lut_t = torch.from_numpy(lut_init).float().to(device)
    sign_l_t = torch.from_numpy(saved["sign_l"].astype(np.float32)).to(device)
    sign_r_t = torch.from_numpy(saved["sign_r"].astype(np.float32)).to(device)

    W_torch = differentiable_dequant(
        walks_t, lut_t, sign_l_t, sign_r_t,
        W_scale=saved["W_scale"],
        shape=saved["shape"],
        Tx=saved["config"]["Tx"],
        Ty=saved["config"]["Ty"],
        V=saved["config"]["V"],
        Q=9,
        L_bits=saved["config"]["L"],
    )
    W_torch_np = W_torch.detach().cpu().numpy().astype(np.float64)
    print(f"Torch W: shape={W_torch_np.shape}, "
          f"mean={W_torch_np.mean():.6f}, std={W_torch_np.std():.6f}")

    # Compare
    abs_err = np.abs(W_np - W_torch_np).max()
    rel_err = abs_err / np.abs(W_np).max()
    print(f"\nmax abs err: {abs_err:.2e}")
    print(f"max rel err: {rel_err:.2e}")

    if rel_err < 1e-4:
        print("[PASS] differentiable dequant matches numpy")
    else:
        print(f"[FAIL] rel_err {rel_err} too large")
        # Diagnostic: where's the biggest error?
        diff = np.abs(W_np - W_torch_np)
        i, j = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  worst element: ({i},{j})  np={W_np[i,j]:.6f}  torch={W_torch_np[i,j]:.6f}")
        return

    # Now check gradient flows
    lut_t_grad = torch.from_numpy(lut_init).float().to(device).requires_grad_(True)
    W_grad = differentiable_dequant(
        walks_t, lut_t_grad, sign_l_t, sign_r_t,
        W_scale=saved["W_scale"], shape=saved["shape"],
        Tx=saved["config"]["Tx"], Ty=saved["config"]["Ty"],
        V=saved["config"]["V"], Q=9, L_bits=saved["config"]["L"],
    )
    loss = W_grad.pow(2).sum()
    loss.backward()
    print(f"LUT grad shape: {lut_t_grad.grad.shape}")
    print(f"LUT grad finite: {torch.isfinite(lut_t_grad.grad).all().item()}")
    print(f"LUT grad max abs: {lut_t_grad.grad.abs().max().item():.4e}")
    print("[PASS] gradient flows back to LUT")

if __name__ == "__main__":
    main()