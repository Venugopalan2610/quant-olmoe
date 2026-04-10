"""Tripwire A.7.1: block LDL decomposition.

A.7.1.1: Synthetic PD matrix, residual < 1e-10 (fp64 Cholesky precision)
A.7.1.2: L has exact Ty x Ty identity diagonal blocks
A.7.1.3: D is block-diagonal (off-diagonal blocks zero), blocks SPD
A.7.1.4: L0 E0 gate_up (hardest Hessian, cond ~10^13) with damping
A.7.1.5: L8 E0 gate_up (well-conditioned) with damping
A.7.1.6: Damping changes cond number as expected

Run: python -m src.tripwires.test_ldl
"""
import os
import sys
import numpy as np
import torch

from src.quantize.ldl import (
    damp_hessian,
    block_ldl,
    block_ldl_residual,
    extract_off_diagonal_A,
)

HESSIAN_DIR = "cache/hessians"


def _identity_diagonal_ok(L, Ty):
    """Check that L has exact Ty x Ty identity blocks on its block diagonal."""
    n = L.shape[0]
    n_blocks = n // Ty
    eye = np.eye(Ty)
    for j in range(n_blocks):
        block = L[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty]
        if not np.allclose(block, eye, atol=1e-10):
            return False, j
    return True, None


def _block_diagonal_ok(D, Ty):
    """Check that D is block-diagonal (off-block entries are zero)."""
    n = D.shape[0]
    n_blocks = n // Ty
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:
                continue
            block = D[i*Ty:(i+1)*Ty, j*Ty:(j+1)*Ty]
            if np.max(np.abs(block)) > 1e-10:
                return False, (i, j)
    return True, None


def _all_D_blocks_spd(D, Ty):
    """Verify each Ty x Ty diagonal block of D is SPD."""
    n = D.shape[0]
    n_blocks = n // Ty
    for j in range(n_blocks):
        block = D[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty]
        # Symmetry
        if not np.allclose(block, block.T, atol=1e-8):
            return False, j, "not symmetric"
        eigs = np.linalg.eigvalsh(block)
        if eigs.min() <= 0:
            return False, j, f"min eig {eigs.min():.2e}"
    return True, None, None


def test_a7_1_1_synthetic():
    """A.7.1.1 — synthetic PD matrix, residual < 1e-10."""
    print("\nA.7.1.1: synthetic PD matrix")
    print("-" * 60)
    Ty = 16
    n = 64  # 4 blocks, small for fast test

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float64)
    H = A @ A.T + n * np.eye(n)  # guaranteed PD

    L, D = block_ldl(H, Ty)
    res = block_ldl_residual(H, L, D)

    id_ok, bad_block = _identity_diagonal_ok(L, Ty)
    bd_ok, bad_pair = _block_diagonal_ok(D, Ty)
    spd_ok, bad_j, bad_reason = _all_D_blocks_spd(D, Ty)

    print(f"  shape: ({n}, {n}), Ty={Ty}")
    print(f"  residual ||LDL^T - H||_F / ||H||_F: {res:.2e}")
    print(f"  [{'PASS' if res < 1e-10 else 'FAIL'}] residual < 1e-10")
    print(f"  [{'PASS' if id_ok else 'FAIL'}] L has identity diagonal blocks"
          + (f" (block {bad_block} bad)" if not id_ok else ""))
    print(f"  [{'PASS' if bd_ok else 'FAIL'}] D is block diagonal"
          + (f" (block {bad_pair} nonzero)" if not bd_ok else ""))
    print(f"  [{'PASS' if spd_ok else 'FAIL'}] D blocks SPD"
          + (f" (block {bad_j}: {bad_reason})" if not spd_ok else ""))

    return res < 1e-10 and id_ok and bd_ok and spd_ok


def test_a7_1_2_easy_real():
    """A.7.1.5 — L8 E0 gate_up, well-conditioned."""
    print("\nA.7.1.5: L8 E0 gate_up (easy, cond ~10^3)")
    print("-" * 60)
    path = os.path.join(HESSIAN_DIR, "L08", "expert_00_gate_up.pt")
    H = torch.load(path, weights_only=True)["H"].numpy().astype(np.float64)

    eigs = np.linalg.eigvalsh(H)
    cond = eigs[-1] / max(eigs[0], 1e-30)
    print(f"  shape: {H.shape}, cond before damping: {cond:.2e}")

    H_damped = damp_hessian(torch.from_numpy(H), damp=0.01).numpy()
    eigs_d = np.linalg.eigvalsh(H_damped)
    cond_d = eigs_d[-1] / max(eigs_d[0], 1e-30)
    print(f"  cond after damping (damp=0.01): {cond_d:.2e}")

    Ty = 16
    L, D = block_ldl(H_damped, Ty)
    res = block_ldl_residual(H_damped, L, D)
    id_ok, _ = _identity_diagonal_ok(L, Ty)
    spd_ok, _, _ = _all_D_blocks_spd(D, Ty)

    print(f"  residual: {res:.2e}")
    print(f"  [{'PASS' if res < 1e-4 else 'FAIL'}] residual < 1e-4")
    print(f"  [{'PASS' if id_ok else 'FAIL'}] L identity diagonal blocks")
    print(f"  [{'PASS' if spd_ok else 'FAIL'}] D blocks SPD")

    return res < 1e-4 and id_ok and spd_ok


def test_a7_1_3_hard_real():
    """A.7.1.4 — L0 E0 gate_up, near-singular (cond ~10^13)."""
    print("\nA.7.1.4: L0 E0 gate_up (hard, cond ~10^13)")
    print("-" * 60)
    path = os.path.join(HESSIAN_DIR, "L00", "expert_00_gate_up.pt")
    H = torch.load(path, weights_only=True)["H"].numpy().astype(np.float64)

    eigs = np.linalg.eigvalsh(H)
    eigs = np.clip(eigs, 1e-30, None)
    cond = eigs[-1] / eigs[0]
    print(f"  shape: {H.shape}, cond before damping: {cond:.2e}")

    # Try without damping first — expected to fail or give huge residual
    try:
        L_raw, D_raw = block_ldl(H, Ty=16)
        res_raw = block_ldl_residual(H, L_raw, D_raw)
        raw_msg = f"succeeded, residual {res_raw:.2e}"
        raw_failed = res_raw > 1e-2
    except np.linalg.LinAlgError as e:
        raw_msg = f"LinAlgError: {e}"
        raw_failed = True
    print(f"  Without damping: {raw_msg}")
    print(f"  [{'PASS' if raw_failed else 'WARN'}] raw LDL unusable (as expected)")

    # With damping
    H_damped = damp_hessian(torch.from_numpy(H), damp=0.01).numpy()
    eigs_d = np.linalg.eigvalsh(H_damped)
    cond_d = eigs_d[-1] / max(eigs_d[0], 1e-30)
    print(f"  cond after damping (damp=0.01): {cond_d:.2e}")

    Ty = 16
    L, D = block_ldl(H_damped, Ty)
    res = block_ldl_residual(H_damped, L, D)
    id_ok, _ = _identity_diagonal_ok(L, Ty)
    spd_ok, bad_j, bad_reason = _all_D_blocks_spd(D, Ty)

    print(f"  residual: {res:.2e}")
    print(f"  [{'PASS' if res < 1e-4 else 'FAIL'}] residual < 1e-4")
    print(f"  [{'PASS' if id_ok else 'FAIL'}] L identity diagonal blocks")
    print(f"  [{'PASS' if spd_ok else 'FAIL'}] D blocks SPD"
          + (f" (block {bad_j}: {bad_reason})" if not spd_ok else ""))

    return res < 1e-4 and id_ok and spd_ok


def test_a7_1_4_damping_effect():
    """A.7.1.6 — damping reduces condition number monotonically."""
    print("\nA.7.1.6: damping effect on condition number")
    print("-" * 60)
    path = os.path.join(HESSIAN_DIR, "L00", "expert_00_gate_up.pt")
    H = torch.load(path, weights_only=True)["H"].numpy().astype(np.float64)

    damps = [0.0, 0.001, 0.01, 0.1]
    conds = []
    for d in damps:
        if d == 0.0:
            Hd = H
        else:
            Hd = damp_hessian(torch.from_numpy(H), damp=d).numpy()
        eigs = np.linalg.eigvalsh(Hd)
        eigs = np.clip(eigs, 1e-30, None)
        cond = eigs[-1] / eigs[0]
        conds.append(cond)
        print(f"  damp={d}: cond={cond:.2e}")

    monotone = all(conds[i] >= conds[i+1] for i in range(len(conds) - 1))
    # damp=0.01 should be < 1e6 (working range for numerical solvers)
    target_ok = conds[2] < 1e6

    print(f"  [{'PASS' if monotone else 'FAIL'}] condition number decreases monotonically")
    print(f"  [{'PASS' if target_ok else 'FAIL'}] damp=0.01 brings cond below 1e6")

    return monotone and target_ok


def test_a7_1_5_A_matrix():
    """A.7.1.7 — extract A = L - I, verify block-strict-lower-triangular."""
    print("\nA.7.1.7: extract A (off-diagonal L)")
    print("-" * 60)
    path = os.path.join(HESSIAN_DIR, "L08", "expert_00_gate_up.pt")
    H = torch.load(path, weights_only=True)["H"].numpy().astype(np.float64)
    H_damped = damp_hessian(torch.from_numpy(H), damp=0.01).numpy()

    Ty = 16
    L, D = block_ldl(H_damped, Ty)
    A = extract_off_diagonal_A(L, Ty)

    n = A.shape[0]
    n_blocks = n // Ty
    diag_zero = True
    for j in range(n_blocks):
        block = A[j*Ty:(j+1)*Ty, j*Ty:(j+1)*Ty]
        if np.max(np.abs(block)) > 1e-10:
            diag_zero = False
            break

    # Also verify A is lower triangular block-wise (upper blocks are zero)
    lower_ok = True
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            block = A[i*Ty:(i+1)*Ty, j*Ty:(j+1)*Ty]
            if np.max(np.abs(block)) > 1e-10:
                lower_ok = False
                break

    # Frobenius norm of A quantifies how much off-diagonal coupling exists
    A_frob = float(np.linalg.norm(A))
    L_frob = float(np.linalg.norm(L))
    print(f"  A shape: {A.shape}")
    print(f"  ||A||_F = {A_frob:.2f}, ||L||_F = {L_frob:.2f}, ratio = {A_frob/L_frob:.3f}")
    print(f"  [{'PASS' if diag_zero else 'FAIL'}] A diagonal blocks are zero")
    print(f"  [{'PASS' if lower_ok else 'FAIL'}] A is block-lower-triangular")

    return diag_zero and lower_ok


def main():
    print("=" * 60)
    print("Tripwire A.7.1: block LDL decomposition")
    print("=" * 60)

    results = []
    results.append(("A.7.1.1 synthetic", test_a7_1_1_synthetic()))
    results.append(("A.7.1.5 easy real (L8)", test_a7_1_2_easy_real()))
    results.append(("A.7.1.4 hard real (L0)", test_a7_1_3_hard_real()))
    results.append(("A.7.1.6 damping effect", test_a7_1_4_damping_effect()))
    results.append(("A.7.1.7 extract A", test_a7_1_5_A_matrix()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.7.1 GATE: PASS — block LDL verified.")
        print("Ready for A.7.2 (BlockLDLQ single-target pipeline).")
        sys.exit(0)
    else:
        print("A.7.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()