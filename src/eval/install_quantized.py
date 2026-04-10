"""Install quantized weights into an OLMoE model.

This is the Phase A reference implementation: dequantize each saved target
to fp16/bf16 and copy it into the model via QuantTarget.set_weight.

Phase C will provide an alternative install path that does NOT dequantize —
instead it replaces nn.Linear modules with kernel-backed modules that
operate directly on bitstreams. The two paths must produce equivalent
forward outputs (verified by Phase C tripwires).
"""
import os
import time
import numpy as np
import torch

from src.codes.ref import decode_hyb_batch
from src.quantize.serialize import load_quantized, dequant_target
from src.models.olmoe_adapter import enumerate_quant_targets

QUANTIZED_DIR = "cache/quantized"
LUT_PATH = "cache/codes/hyb_lut_init.npy"


def _make_decoder():
    """The HYB V=2 decoder, matching what was used at quantization time."""
    lut = np.load(LUT_PATH)
    def hyb_v2(states):
        return decode_hyb_batch(states, lut, Q=9)
    return hyb_v2


def _quantized_path_for_target(target, quant_dir):
    """Map a QuantTarget to its saved file path under quant_dir/L{NN}/."""
    L = target.layer_idx
    if target.kind == "attention":
        fname = f"attn_{target.proj}.pt"
    else:
        fname = f"expert_{target.expert_idx:02d}_{target.proj}.pt"
    return os.path.join(quant_dir, f"L{L:02d}", fname)


def install_quantized_weights(model, quant_dir, verbose=True):
    """Dequantize every saved target and install it into the model.

    Operates in-place on `model`. After return, the model holds dequantized
    bf16 weights derived from the 2-bit quantized payloads.

    Args:
        model: an OlmoeForCausalLM (typically loaded fresh in bf16)
        quant_dir: path to per-target .pt files
        verbose: progress prints

    Returns:
        stats: dict with counts and timing
    """
    decode = _make_decoder()
    targets = list(enumerate_quant_targets(model))

    if verbose:
        print(f"Installing {len(targets)} quantized targets from {quant_dir}")

    n_installed = 0
    n_missing = 0
    t0 = time.time()
    last_print = t0

    for target in targets:
        path = _quantized_path_for_target(target, quant_dir)  # ← pass quant_dir
        if not os.path.exists(path):
            n_missing += 1
            if verbose:
                print(f"  MISSING: {target.name} -> {path}")
            continue

        saved = load_quantized(path)
        Wh_fp64 = dequant_target(saved, decode)
        Wh_torch = torch.from_numpy(Wh_fp64.astype(np.float32))
        target.set_weight(Wh_torch)
        n_installed += 1

        if verbose and time.time() - last_print > 5.0:
            elapsed = time.time() - t0
            rate = n_installed / elapsed
            eta = (len(targets) - n_installed) / max(rate, 1e-6)
            print(f"  {n_installed}/{len(targets)} installed "
                  f"({rate:.0f}/s, ETA {eta:.0f}s)")
            last_print = time.time()

    elapsed = time.time() - t0
    if verbose:
        print(f"  Done: {n_installed}/{len(targets)} installed in {elapsed:.0f}s")
        if n_missing:
            print(f"  WARNING: {n_missing} targets had no saved file")

    return {
        "n_installed": n_installed,
        "n_missing": n_missing,
        "n_targets": len(targets),
        "elapsed_seconds": elapsed,
    }