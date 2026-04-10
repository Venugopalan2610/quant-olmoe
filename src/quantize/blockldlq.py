"""BlockLDLQ reference implementation.

When return_bitstreams=True, also returns the trellis bitstreams + start_states
per col block, suitable for storing in the quantized format and dequanting later.
"""
import numpy as np
import torch

from src.rht.transform import apply_rht, apply_inverse_rht
from src.quantize.ldl import damp_hessian, block_ldl, extract_off_diagonal_A
from src.viterbi.encode import viterbi_encode_v, precompute_codebook_v


def blockldlq(
    W, H, sign_l, sign_r, decode_fn,
    L_bits=16, k=2, V=2, Tx=16, Ty=16, damp=0.01,
    use_cuda=False,
    return_diagnostics=False,
    return_bitstreams=False,
):
    """Quantize one weight matrix with BlockLDLQ.

    Returns:
        Wh, Wh_tilde, proxy_loss
        + diagnostics if return_diagnostics
        + bitstreams_dict if return_bitstreams (contains 'bitstreams' and 'start_states')
    """
    m, n = W.shape
    assert n % Ty == 0
    assert m % Tx == 0

    W64 = W.astype(np.float64)
    H64 = H.astype(np.float64)

    # RHT
    W_tilde = apply_rht(W64, sign_l, sign_r)
    H_tilde = apply_rht(H64, sign_r, sign_r)

    # Scale normalization
    W_scale = float(np.sqrt((W_tilde ** 2).mean()))
    if W_scale < 1e-30:
        W_scale = 1.0
    W_tilde_unit = W_tilde / W_scale

    # Damp + LDL
    H_damped = damp_hessian(torch.from_numpy(H_tilde), damp=damp).numpy()
    L_mat, D_mat = block_ldl(H_damped, Ty)
    A = extract_off_diagonal_A(L_mat, Ty)

    # CUDA setup
    if use_cuda:
        from src.viterbi.cuda_kernel import viterbi_encode_v_batched_cuda
        codebook_np = precompute_codebook_v(L_bits, V, decode_fn)
        codebook_gpu = torch.from_numpy(codebook_np).cuda().contiguous()

    Wh_tilde_unit = np.zeros_like(W_tilde_unit)
    n_col_blocks = n // Ty
    n_row_blocks = m // Tx
    T_seq = Tx * Ty
    n_steps = T_seq // V
    scale_sq = W_scale * W_scale

    diagnostics = {
        "proxy_loss_per_step": [],
        "n_viterbi_calls": 0,
        "per_block_mse_in_tile": [],
        "W_scale": W_scale,
    }

    if return_bitstreams:
        all_bitstreams = np.zeros((n_col_blocks, n_row_blocks, n_steps), dtype=np.uint8)
        all_start_states = np.zeros((n_col_blocks, n_row_blocks), dtype=np.int32)

    for j in range(n_col_blocks - 1, -1, -1):
        col_start = j * Ty
        col_end = col_start + Ty

        error = W_tilde_unit - Wh_tilde_unit
        feedback = error @ A[:, col_start:col_end]
        x_block = W_tilde_unit[:, col_start:col_end] + feedback

        x_reshaped = x_block.reshape(n_row_blocks, T_seq).astype(np.float32)

        if use_cuda:
            x_gpu = torch.from_numpy(x_reshaped).cuda()
            bs_gpu, ss_gpu, recons_gpu, mses_gpu = viterbi_encode_v_batched_cuda(
                x_gpu, codebook_gpu,
            )
            xhat_reshaped = recons_gpu.cpu().numpy().astype(np.float64)
            block_mse_sum = float(mses_gpu.sum().cpu())
            diagnostics["n_viterbi_calls"] += n_row_blocks
            if return_bitstreams:
                bs_np = bs_gpu.cpu().numpy()
                all_bitstreams[j] = bs_np.astype(np.uint8)
                all_start_states[j] = ss_gpu.cpu().numpy().astype(np.int32)
        else:
            xhat_reshaped = np.zeros_like(x_reshaped, dtype=np.float64)
            block_mse_sum = 0.0
            for i in range(n_row_blocks):
                bs, ss, recon, mse = viterbi_encode_v(
                    x_reshaped[i], L=L_bits, k=k, V=V, decode_fn=decode_fn,
                )
                xhat_reshaped[i] = recon.astype(np.float64)
                block_mse_sum += mse
                diagnostics["n_viterbi_calls"] += 1
                if return_bitstreams:
                    all_bitstreams[j, i] = bs.astype(np.uint8)
                    all_start_states[j, i] = ss

        Wh_tilde_unit[:, col_start:col_end] = xhat_reshaped.reshape(m, Ty)
        diagnostics["per_block_mse_in_tile"].append(block_mse_sum / n_row_blocks)

        if return_diagnostics:
            diff_unit = Wh_tilde_unit - W_tilde_unit
            current_loss_unit = float(np.trace(diff_unit @ H_tilde @ diff_unit.T) / m)
            diagnostics["proxy_loss_per_step"].append(current_loss_unit * scale_sq)

    Wh_tilde = Wh_tilde_unit * W_scale
    diff_final = Wh_tilde - W_tilde
    proxy_loss = float(np.trace(diff_final @ H_tilde @ diff_final.T) / m)
    Wh = apply_inverse_rht(Wh_tilde, sign_l, sign_r)

    out = [Wh, Wh_tilde, proxy_loss]
    if return_diagnostics:
        out.append(diagnostics)
    if return_bitstreams:
        out.append({
            "bitstreams": all_bitstreams,
            "start_states": all_start_states,
            "W_scale": W_scale,
        })
    return tuple(out) if (return_diagnostics or return_bitstreams) else (Wh, Wh_tilde, proxy_loss)