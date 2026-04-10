"""Random Hadamard Transform for incoherence processing.

RHT(W) = (1/sqrt(n)) * H_n * S_n * W * S_m^T * H_m^T  (outer view)
       = sandwich W between sign-flipped Hadamards on both sides

Uses fast-hadamard-transform if available, falls back to a numpy FWHT.

Crucial: the same sign vectors must be used for forward and inverse, and
they must be saved alongside the quantized weights for inference-time
inverse RHT on activations.
"""
import numpy as np
import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fht_cuda
    _HAS_FAST_FHT = True
except ImportError:
    _HAS_FAST_FHT = False


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def fwht_numpy(x):
    """In-place Fast Walsh-Hadamard Transform on the last axis of a numpy array.

    Result is unscaled (no 1/sqrt(n)). Length must be a power of 2.
    """
    x = x.copy().astype(np.float32)
    n = x.shape[-1]
    assert _is_pow2(n), f"FWHT length must be power of 2, got {n}"
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i:i + h].copy()
            b = x[..., i + h:i + 2 * h].copy()
            x[..., i:i + h] = a + b
            x[..., i + h:i + 2 * h] = a - b
        h *= 2
    return x


def fht(x):
    """Fast Hadamard Transform on the last axis (unscaled).

    Uses fast_hadamard_transform CUDA kernel if x is a CUDA tensor and the
    package is available; otherwise falls back to numpy FWHT.
    """
    if isinstance(x, torch.Tensor):
        if _HAS_FAST_FHT and x.is_cuda:
            return _fht_cuda(x)
        # CPU torch tensor: go through numpy
        return torch.from_numpy(fwht_numpy(x.cpu().numpy())).to(x.device).to(x.dtype)
    return fwht_numpy(x)


def make_sign_vector(n, seed):
    """Random ±1 vector of length n."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n).astype(np.float32)


def apply_rht(W, sign_left, sign_right):
    """Apply RHT to a weight matrix W of shape (m, n).

    W̃ = (1/sqrt(m*n)) * H_m * diag(sign_left) * W * diag(sign_right) * H_n

    Both m and n must be powers of 2.
    """
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().float().numpy()
        was_torch = True
        device, dtype = W.device, W.dtype
    else:
        W_np = W.astype(np.float32)
        was_torch = False

    m, n = W_np.shape
    assert _is_pow2(m) and _is_pow2(n), f"RHT requires power-of-2 dims, got {(m, n)}"

    # Right side: W * diag(sign_right) * H_n
    Wr = W_np * sign_right[None, :]
    Wr = fwht_numpy(Wr)  # FWHT along last axis

    # Left side: H_m * diag(sign_left) * Wr
    Wl = (Wr * sign_left[:, None]).T  # transpose so FWHT acts on m-dim
    Wl = fwht_numpy(Wl)
    Wl = Wl.T

    # Normalization
    Wl = Wl / np.sqrt(m * n)

    if was_torch:
        return torch.from_numpy(Wl).to(device=device, dtype=dtype)
    return Wl


def apply_inverse_rht(W_tilde, sign_left, sign_right):
    """Inverse of apply_rht. Hadamard is self-inverse up to scaling, so:

    W = (1/sqrt(m*n)) * diag(sign_left) * H_m * W̃ * H_n * diag(sign_right)
    """
    if isinstance(W_tilde, torch.Tensor):
        W_np = W_tilde.detach().cpu().float().numpy()
        was_torch = True
        device, dtype = W_tilde.device, W_tilde.dtype
    else:
        W_np = W_tilde.astype(np.float32)
        was_torch = False

    m, n = W_np.shape

    # Right: W̃ * H_n * diag(sign_right)
    Wr = fwht_numpy(W_np)
    Wr = Wr * sign_right[None, :]

    # Left: diag(sign_left) * H_m * Wr
    Wl = fwht_numpy(Wr.T).T
    Wl = Wl * sign_left[:, None]

    Wl = Wl / np.sqrt(m * n)

    if was_torch:
        return torch.from_numpy(Wl).to(device=device, dtype=dtype)
    return Wl