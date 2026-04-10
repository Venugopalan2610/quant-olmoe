"""Smoke test ft_one_expert with synthetic activations.

Generates X randomly, computes Y as the fp32 forward through the loaded
quantized expert (with the GLOBAL initial LUT). FT should drive loss
toward zero because the model is exactly representable with the init LUT.
This validates the autograd chain end-to-end before we plug in real
activations.
"""
import json
import numpy as np
import torch
from safetensors import safe_open

from src.quantize.serialize import load_quantized
from src.finetune.quant_expert import QuantizedExpert, ft_one_expert

LAYER = 8
EXPERT = 0
QUANT_DIR = f"cache/quantized/L{LAYER:02d}"
LUT_PATH = "cache/codes/hyb_lut_init.npy"
DEVICE = "cuda:0"


def main():
    # Load three payloads
    gate_p = load_quantized(f"{QUANT_DIR}/expert_{EXPERT:02d}_gate_proj.pt")
    up_p   = load_quantized(f"{QUANT_DIR}/expert_{EXPERT:02d}_up_proj.pt")
    down_p = load_quantized(f"{QUANT_DIR}/expert_{EXPERT:02d}_down_proj.pt")
    lut_init = np.load(LUT_PATH)

    print(f"Loaded L{LAYER} expert {EXPERT}")
    print(f"  gate shape: {gate_p['shape']}, up shape: {up_p['shape']}, down shape: {down_p['shape']}")

    # Build a "ground truth" expert with the init LUT
    gt_expert = QuantizedExpert(gate_p, up_p, down_p, lut_init, device=DEVICE)
    gt_expert.eval()

    # Generate synthetic input — match expected hidden_size from gate shape
    # gate is (intermediate, hidden) so input dim is gate.shape[1]
    n_tokens = 1024
    hidden_in = gate_p["shape"][1]
    print(f"  hidden_in: {hidden_in}")

    torch.manual_seed(42)
    X = torch.randn(n_tokens, hidden_in, device=DEVICE, dtype=torch.float32) * 0.5

    # Y = fp32 forward of GT expert. Note: with init LUT, GT expert IS what
    # we're trying to recover. So FT initialized at lut_init should already
    # have loss ≈ 0. To make this a meaningful test, we PERTURB the init LUT
    # for the trainable expert and check that FT recovers it.
    with torch.no_grad():
        Y = gt_expert(X)

    # Perturb the init LUT for the trainable expert
    lut_perturbed = lut_init + 0.05 * np.random.randn(*lut_init.shape).astype(np.float32)

    print(f"\nFine-tuning with perturbed LUT init (added 0.05 std noise)")
    result = ft_one_expert(
        gate_p, up_p, down_p, lut_perturbed,
        X, Y,
        n_steps=500, lr=5e-4, device=DEVICE, verbose=True,
    )

    print(f"\n  loss_init:    {result['loss_init']:.6e}")
    print(f"  loss_final:   {result['loss_final']:.6e}")
    print(f"  improvement:  {result['improvement']*100:.1f}%")

    # Check LUTs moved toward init (the "true" answer)
    drift_gate = np.abs(result['lut_gate'] - lut_init).mean()
    drift_init = np.abs(lut_perturbed - lut_init).mean()
    print(f"\n  Gate LUT drift from truth: init={drift_init:.4f} -> final={drift_gate:.4f}")

    if result['improvement'] > 0.5:
        print("\n[PASS] FT reduces loss by >50%, autograd chain works")
    else:
        print(f"\n[FAIL] FT only reduced loss by {result['improvement']*100:.1f}% — investigate")


if __name__ == "__main__":
    main()