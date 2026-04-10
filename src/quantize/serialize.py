"""Save/load quantized weight format and dequantization helper.

The on-disk format is one .pt file per QuantTarget containing the BlockLDLQ
output (bitstreams + RHT signs + scale + metadata) sufficient to reconstruct
the dequantized weight.

Disk size per target: bitstreams dominate at uint8 (one nibble per byte).
True 4-bits-per-byte packing is Phase C work.
"""
import os
import numpy as np
import torch

from src.rht.transform import apply_inverse_rht
from src.viterbi.encode import precompute_codebook_v


def save_quantized(path, bitstreams, start_states, sign_l, sign_r, W_scale,
                   shape, meta, config=None):
    """Write a quantized target to disk.

    Args:
        path: output .pt path
        bitstreams: (n_col_blocks, n_row_blocks, n_steps) uint8 (values 0-15)
        start_states: (n_col_blocks, n_row_blocks) int32
        sign_l: (m,) ±1 vector (any numeric type)
        sign_r: (n,) ±1 vector
        W_scale: float
        shape: (m, n)
        meta: dict with at least {layer, kind, proj} and optional expert
        config: dict with L, k, V, Tx, Ty (defaults to HYB config)
    """
    if config is None:
        config = {"L": 16, "k": 2, "V": 2, "Tx": 16, "Ty": 16}

    payload = {
        "bitstreams": np.ascontiguousarray(bitstreams, dtype=np.uint8),
        "start_states": np.ascontiguousarray(start_states, dtype=np.int32),
        "sign_l": np.sign(np.asarray(sign_l)).astype(np.int8),
        "sign_r": np.sign(np.asarray(sign_r)).astype(np.int8),
        "W_scale": float(W_scale),
        "shape": tuple(shape),
        "meta": meta,
        "config": config,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_quantized(path):
    """Load a quantized target file. Returns the payload dict."""
    return torch.load(path, weights_only=False)


def dequant_target(saved, decode_fn):
    """Reconstruct Wh (the dequantized weight) from a saved quantized payload.

    Args:
        saved: payload dict from load_quantized()
        decode_fn: codebook lookup function (uint32 array -> (N, V) float)
                   This must be the SAME decoder used at quantization time.

    Returns:
        Wh: (m, n) float64 numpy array, the dequantized weight in original basis
    """
    bitstreams = saved["bitstreams"]            # (n_col_blocks, n_row_blocks, n_steps) uint8
    start_states = saved["start_states"]        # (n_col_blocks, n_row_blocks) int32
    sign_l = saved["sign_l"].astype(np.float32)
    sign_r = saved["sign_r"].astype(np.float32)
    W_scale = float(saved["W_scale"])
    m, n = saved["shape"]
    cfg = saved["config"]
    L_bits, k, V, Tx, Ty = cfg["L"], cfg["k"], cfg["V"], cfg["Tx"], cfg["Ty"]

    n_col_blocks, n_row_blocks, n_steps = bitstreams.shape
    assert n_col_blocks == n // Ty
    assert n_row_blocks == m // Tx
    assert n_steps == (Tx * Ty) // V

    codebook = precompute_codebook_v(L_bits, V, decode_fn)  # (2^L, V) float32

    Wh_tilde_unit = np.zeros((m, n), dtype=np.float64)
    kV = k * V
    mask = (1 << L_bits) - 1
    T_seq = Tx * Ty

    for j in range(n_col_blocks):
        col_start = j * Ty
        col_end = col_start + Ty

        bs_block = bitstreams[j].astype(np.int64)        # (n_row_blocks, n_steps)
        ss_block = start_states[j].astype(np.int64)      # (n_row_blocks,)

        # Replay walks: same logic as viterbi_decode_v but batched
        s = ss_block.copy()
        states_walk = np.zeros_like(bs_block)
        for t in range(n_steps):
            s = ((s << kV) | bs_block[:, t]) & mask
            states_walk[:, t] = s

        # Codebook lookup -> (n_row_blocks, n_steps, V)
        recons_steps = codebook[states_walk]
        recons_flat = recons_steps.reshape(n_row_blocks, T_seq).astype(np.float64)
        Wh_tilde_unit[:, col_start:col_end] = recons_flat.reshape(m, Ty)

    Wh_tilde = Wh_tilde_unit * W_scale
    Wh = apply_inverse_rht(Wh_tilde, sign_l, sign_r)
    return Wh