"""Differentiable dequantization for LUT fine-tuning.

Trellis state walks are precomputed in numpy and frozen.
LUT gather is done in torch with gradients flowing back.
Inverse RHT is differentiable via fast_hadamard_transform.
"""
import numpy as np
import torch
import torch.nn as nn

try:
    from fast_hadamard_transform import hadamard_transform as _fht_cuda
    _HAS_FAST_FHT = True
except ImportError:
    _HAS_FAST_FHT = False


# ============================================================================
# Inverse RHT (torch, differentiable) — matches src.rht.transform
# ============================================================================

def _fht_torch(x):
    """FHT on last axis. CUDA via fast_hadamard_transform package, else fallback.

    Differentiable: FHT is an orthogonal linear op so autograd handles it.
    """
    if _HAS_FAST_FHT and x.is_cuda:
        return _fht_cuda(x)
    # CPU fallback
    n = x.shape[-1]
    h = 1
    while h < n:
        x_reshaped = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_reshaped[..., 0, :].clone()
        b = x_reshaped[..., 1, :].clone()
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return x


def inverse_rht_torch(W_tilde, sign_l, sign_r):
    """Differentiable inverse RHT matching src.rht.transform.apply_inverse_rht.

    W = (1/sqrt(m*n)) * diag(sign_l) * H_m * W_tilde * H_n * diag(sign_r)
    """
    m, n = W_tilde.shape
    Wr = _fht_torch(W_tilde)
    Wr = Wr * sign_r.unsqueeze(0)
    Wl = _fht_torch(Wr.t().contiguous()).t().contiguous()
    Wl = Wl * sign_l.unsqueeze(1)
    Wl = Wl / (float(m) * float(n)) ** 0.5
    return Wl


# ============================================================================
# Trellis walk replay (numpy, run once per target)
# ============================================================================

def precompute_walk_states(bitstreams, start_states, L_bits=16, kV=4):
    """Replay trellis walks. Pure numpy. Match dequant_target() in serialize.py.

    Args:
        bitstreams: (n_col_blocks, n_row_blocks, n_steps) uint8, values 0-15
        start_states: (n_col_blocks, n_row_blocks) int32
    Returns:
        walks: (n_col_blocks, n_row_blocks, n_steps) int64, state at each step
    """
    bs = bitstreams.astype(np.int64)
    ss = start_states.astype(np.int64)
    n_col, n_row, n_steps = bs.shape
    mask = (1 << L_bits) - 1

    walks = np.zeros_like(bs)
    for j in range(n_col):
        s = ss[j].copy()  # (n_row,)
        for t in range(n_steps):
            s = ((s << kV) | bs[j, :, t]) & mask
            walks[j, :, t] = s
    return walks


# ============================================================================
# Differentiable codebook (torch port of decode_hyb_batch)
# ============================================================================

def build_differentiable_codebook(lut, Q=9, L_bits=16):
    """Materialize the full (2^L, V) codebook from a (2^Q, V) trainable LUT.

    Replicates decode_hyb_batch in torch:
        x = state * state + state  (mod 2^32)
        idx = (x >> (15 - Q)) & ((1 << Q) - 1)
        out[0] = lut[idx, 0]
        out[1] = lut[idx, 1] * (-1 if (x >> 15) & 1 else 1)
    """
    device = lut.device
    n_states = 1 << L_bits  # 65536

    states = torch.arange(n_states, dtype=torch.int64, device=device)
    x = (states * states + states) & 0xFFFFFFFF
    idx = ((x >> (15 - Q)) & ((1 << Q) - 1)).long()
    sign_flip = ((x >> 15) & 1).bool()
    sign_factor = torch.where(sign_flip, -1.0, 1.0).to(lut.dtype)

    col0 = lut[idx, 0]
    col1 = lut[idx, 1] * sign_factor
    codebook = torch.stack([col0, col1], dim=-1)  # (2^L, 2)
    return codebook


# ============================================================================
# Full differentiable dequant
# ============================================================================

def differentiable_dequant(walk_states, lut, sign_l, sign_r, W_scale,
                            shape, Tx=16, Ty=16, V=2, Q=9, L_bits=16):
    """Differentiable W = dequant(walk_states, lut, signs, scale).

    Matches dequant_target() in serialize.py exactly. The reshape pattern is
    the critical part — must lay out (n_row_blocks, n_steps, V) tile sequences
    into (m, Ty) blocks correctly.
    """
    m, n = shape
    n_col_blocks, n_row_blocks, n_steps = walk_states.shape
    T_seq = Tx * Ty  # 256
    assert n_col_blocks == n // Ty, f"col blocks: {n_col_blocks} vs {n//Ty}"
    assert n_row_blocks == m // Tx, f"row blocks: {n_row_blocks} vs {m//Tx}"
    assert n_steps == T_seq // V, f"n_steps: {n_steps} vs {T_seq//V}"

    codebook = build_differentiable_codebook(lut, Q=Q, L_bits=L_bits)

    # Gather codebook entries: (n_col, n_row, n_steps, V)
    recons = codebook[walk_states]

    # Build W_unit by laying out each column block
    # For each j in n_col: block is (n_row_blocks, n_steps, V)
    # Flatten to (n_row_blocks, T_seq), then reshape to (m, Ty)
    cols = []
    for j in range(n_col_blocks):
        block = recons[j]  # (n_row, n_steps, V)
        flat = block.reshape(n_row_blocks, T_seq)  # (n_row, 256)
        col_block = flat.reshape(m, Ty)  # (m, Ty)
        cols.append(col_block)
    W_unit = torch.cat(cols, dim=1)  # (m, n)

    # Apply scale
    W_tilde = W_unit * W_scale

    # Inverse RHT
    W_dequant = inverse_rht_torch(W_tilde, sign_l, sign_r)

    return W_dequant