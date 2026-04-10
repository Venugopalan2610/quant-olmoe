"""Block LDL decomposition for BlockLDLQ.

Factorizes a symmetric PD matrix H into H = L D L^T where:
  - L is block-lower-triangular with Ty x Ty IDENTITY diagonal blocks
  - D is block-diagonal with Ty x Ty SPD blocks

This is a different factorization from scalar LDL (which has scalar 1s on
the L diagonal and a diagonal D). Block LDL matches BlockLDLQ's feedback
structure: each Ty x Ty block of columns forms one quantization tile, and
the off-diagonal L blocks encode how to propagate block-level errors.

The damping function adds `damp * mean(diag(H)) * I` before factorization,
which is the standard GPTQ/QuIP#/QTIP regularizer for near-singular H.
Critical for OLMoE layers 0-2 where raw condition numbers reach ~10^13.
"""
import numpy as np
import torch


def damp_hessian(H, damp=0.01):
    """Add damp * mean(diag(H)) * I to stabilize LDL.

    H: (n, n) tensor or array, symmetric
    Returns: damped H of same type/shape

    With damp=0.01, this regularizes toward the average diagonal scale,
    which is mild enough that well-conditioned Hs are essentially unchanged
    but ill-conditioned Hs (cond > 1e8) become tractable.
    """
    if isinstance(H, torch.Tensor):
        diag_mean = H.diagonal().mean()
        n = H.shape[0]
        eye = torch.eye(n, dtype=H.dtype, device=H.device)
        return H + damp * diag_mean * eye
    else:
        n = H.shape[0]
        diag_mean = np.mean(np.diagonal(H))
        return H + damp * diag_mean * np.eye(n, dtype=H.dtype)


def block_ldl(H, Ty):
    """Block LDL factorization: H = L D L^T.

    Args:
        H: (n, n) numpy float32/float64, symmetric PD. n must be divisible by Ty.
        Ty: block size (e.g. 16 for QTIP HYB config).

    Returns:
        L: (n, n) block-lower-triangular with Ty x Ty identity diagonal blocks
        D: (n, n) block-diagonal with Ty x Ty SPD blocks (dense storage, most
           entries zero)

    Implementation: run scipy Cholesky, then derive block-LDL from the
    block structure of the Cholesky factor. See module docstring for math.

    Raises numpy.linalg.LinAlgError if H is not PD (damp it first!).
    """
    n = H.shape[0]
    assert n % Ty == 0, f"matrix size {n} not divisible by block size {Ty}"
    n_blocks = n // Ty

    # Use numpy cholesky for reliability; result is lower triangular
    L_chol = np.linalg.cholesky(H.astype(np.float64))  # more precision for ill-cond

    L = np.zeros_like(L_chol)
    D = np.zeros_like(L_chol)
    eye_Ty = np.eye(Ty, dtype=L_chol.dtype)

    # Identity on L diagonal blocks
    for j in range(n_blocks):
        L[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty] = eye_Ty

    # For each block column j:
    #   T_j = diagonal block of L_chol (Ty x Ty lower triangular)
    #   D[j,j] = T_j @ T_j.T
    #   L[i,j] = L_chol[i,j] @ T_j^{-1}  for i > j
    for j in range(n_blocks):
        T_j = L_chol[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty]  # lower triangular
        D[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty] = T_j @ T_j.T

        if j + 1 < n_blocks:
            # Block below-diagonal column: rows [(j+1)*Ty : n], cols [j*Ty : (j+1)*Ty]
            below = L_chol[(j+1)*Ty:, j*Ty:(j+1)*Ty]  # ((n - (j+1)*Ty), Ty)
            # We need below @ T_j^{-1}. Since T_j is lower triangular, solve
            # (T_j.T @ X.T = below.T) and transpose.
            from scipy.linalg import solve_triangular
            inv_block = solve_triangular(
                T_j, eye_Ty, lower=True, unit_diagonal=False,
            )  # T_j^{-1}
            L[(j+1)*Ty:, j*Ty:(j+1)*Ty] = below @ inv_block

    return L.astype(H.dtype), D.astype(H.dtype)


def block_ldl_residual(H, L, D):
    """Return ||L D L^T - H||_F / ||H||_F for tripwire verification."""
    reconstructed = L @ D @ L.T
    diff = reconstructed - H
    return float(np.linalg.norm(diff) / (np.linalg.norm(H) + 1e-30))


def extract_off_diagonal_A(L, Ty):
    """Return A = L - I (block-strictly-lower-triangular) for BlockLDLQ feedback.

    In BlockLDLQ notation, the error feedback is driven by A where L = I + A,
    so A is block-lower-triangular with ZERO diagonal blocks. The feedback
    at block j uses A's blocks below the (j,j) position.
    """
    n = L.shape[0]
    n_blocks = n // Ty
    A = L.copy()
    for j in range(n_blocks):
        A[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty] = 0.0
    return A