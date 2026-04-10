"""Bitshift-trellis Viterbi encoder, V=1 reference implementation.

Right-to-left dynamic programming over a length-T sequence. Tracks the
best partial walk ending at every possible L-bit state, with backpointers
for traceback. Tail-biting handled separately by Algorithm 4.

Notation matches the QTIP paper:
  L = state width in bits
  k = bits emitted per step (= bits per weight at V=1)
  V = vector dimension (this file: V=1 only)
  T = sequence length

For V=1:
  - 2^L total states
  - Each state has 2^k successors (and 2^k predecessors)
  - Bitshift rule: next = ((s << k) | c) & mask, c in [0, 2^k)
  - Each step decodes 1 weight: w_hat = decode_fn(state)
"""
import numpy as np


def precompute_predecessors(L, k):
    """For each state s in [0, 2^L), list its 2^k predecessors.

    A state s' is a predecessor of s iff there exists c in [0, 2^k) such that
    s = ((s' << k) | c) & mask. Solving: s's top (L-k) bits = s's predecessor's
    bottom (L-k) bits, and the predecessor's top k bits can be anything.
    """
    n_states = 1 << L
    n_pred = 1 << k

    preds = np.zeros((n_states, n_pred), dtype=np.int32)
    for s in range(n_states):
        top = s >> k  # s's top (L-k) bits
        for p_top in range(n_pred):
            preds[s, p_top] = (p_top << (L - k)) | top
    return preds


def precompute_codebook(L, decode_fn):
    """Materialize decode_fn(s) for every state s in [0, 2^L)."""
    n_states = 1 << L
    states = np.arange(n_states, dtype=np.uint32)
    return decode_fn(states).astype(np.float32)


def viterbi_encode(sequence, L, k, decode_fn, return_trace=False):
    """Quantize a length-T sequence with the bitshift trellis (V=1)."""
    T = len(sequence)
    n_states = 1 << L

    codebook = precompute_codebook(L, decode_fn)
    preds = precompute_predecessors(L, k)

    cum_err = np.zeros(n_states, dtype=np.float32)
    backpointers = np.zeros((T, n_states), dtype=np.int32)

    if return_trace:
        trace_err = np.zeros((T, n_states), dtype=np.float32)

    for t in range(T):
        w_t = sequence[t]
        local_cost = (codebook - w_t) ** 2
        pred_costs = cum_err[preds]
        new_cum = pred_costs.min(axis=1) + local_cost
        best_pred_idx = pred_costs.argmin(axis=1)
        backpointers[t] = preds[np.arange(n_states), best_pred_idx]
        cum_err = new_cum
        if return_trace:
            trace_err[t] = cum_err

    final_state = int(cum_err.argmin())

    states_walk = np.zeros(T, dtype=np.int32)
    states_walk[T - 1] = final_state
    for t in range(T - 1, 0, -1):
        states_walk[t - 1] = backpointers[t, states_walk[t]]
    start_state = int(backpointers[0, states_walk[0]])

    mask_k = (1 << k) - 1
    bitstream = (states_walk & mask_k).astype(np.int32)
    recon = codebook[states_walk]
    mse = float(np.mean((recon - sequence) ** 2))

    if return_trace:
        return bitstream, start_state, recon, mse, trace_err, backpointers
    return bitstream, start_state, recon, mse


def viterbi_decode(bitstream, start_state, L, k, decode_fn):
    """Replay a walk from the bitstream and start_state, return decoded values."""
    T = len(bitstream)
    mask = (1 << L) - 1
    mask_k = (1 << k) - 1

    states = np.zeros(T, dtype=np.uint32)
    s = np.uint32(start_state)
    for t in range(T):
        s = ((s << np.uint32(k)) | np.uint32(bitstream[t] & mask_k)) & np.uint32(mask)
        states[t] = s

    return decode_fn(states).astype(np.float32)


def viterbi_encode_constrained(sequence, L, k, decode_fn, overlap_bits, overlap_value):
    """Viterbi with start/end state constraint on s_0 and s_{T-1}.

    s_0 (the state decoding the first weight) must have its top `overlap_bits`
    bits equal to overlap_value.

    s_{T-1} (the state decoding the last weight) must have its bottom
    `overlap_bits` bits equal to overlap_value.

    Together these enforce a valid wrap-around s_{T-1} -> s_0 in the bitshift
    trellis: s_0.top(L-k) == s_{T-1}.bottom(L-k) == overlap_value.
    """
    T = len(sequence)
    n_states = 1 << L

    codebook = precompute_codebook(L, decode_fn)
    preds = precompute_predecessors(L, k)

    INF = np.float32(1e18)

    cum_err = np.zeros(n_states, dtype=np.float32)
    backpointers = np.zeros((T, n_states), dtype=np.int32)

    # Mask for the s_0 constraint: top `overlap_bits` bits must equal overlap_value
    top_mask = ((1 << overlap_bits) - 1) << (L - overlap_bits)
    target_top = (overlap_value & ((1 << overlap_bits) - 1)) << (L - overlap_bits)
    valid_s0 = (np.arange(n_states) & top_mask) == target_top

    for t in range(T):
        w_t = sequence[t]
        local_cost = (codebook - w_t) ** 2
        pred_costs = cum_err[preds]
        new_cum = pred_costs.min(axis=1) + local_cost
        best_pred_idx = pred_costs.argmin(axis=1)
        backpointers[t] = preds[np.arange(n_states), best_pred_idx]

        if t == 0:
            # Apply start constraint to s_0
            new_cum = np.where(valid_s0, new_cum, INF)

        cum_err = new_cum

    # End constraint: s_{T-1}.bottom(overlap_bits) == overlap_value
    bottom_mask = (1 << overlap_bits) - 1
    target_bottom = overlap_value & bottom_mask
    valid_ends = (np.arange(n_states) & bottom_mask) == target_bottom

    masked_err = np.where(valid_ends, cum_err, INF)
    final_state = int(masked_err.argmin())
    final_err = float(masked_err[final_state])

    if final_err >= INF / 2:
        raise RuntimeError(
            f"No valid tail-biting walk found for overlap_value={overlap_value}."
        )

    states_walk = np.zeros(T, dtype=np.int32)
    states_walk[T - 1] = final_state
    for t in range(T - 1, 0, -1):
        states_walk[t - 1] = backpointers[t, states_walk[t]]
    start_state = int(backpointers[0, states_walk[0]])

    mask_k = (1 << k) - 1
    bitstream = (states_walk & mask_k).astype(np.int32)
    recon = codebook[states_walk]
    mse = float(np.mean((recon - sequence) ** 2))

    return bitstream, start_state, recon, mse


def viterbi_encode_tailbiting(sequence, L, k, decode_fn):
    """Tail-biting Viterbi via Algorithm 4 (paper Section 3.2).

    1. Rotate sequence right by T/2.
    2. Run unconstrained Viterbi on the rotated sequence.
    3. Replay the rotated walk to find state[T/2]. Its top (L-k) bits are
       the overlap with state[T/2 - 1] under the bitshift rule.
    4. Re-encode the original sequence with that overlap as the start/end
       constraint via viterbi_encode_constrained.
    """
    T = len(sequence)
    overlap_bits = L - k

    # Step 1+2: rotate right by T/2 and run unconstrained Viterbi
    rotated = np.roll(sequence, T // 2)
    bs_rot, start_rot, _recon_rot, _mse_rot = viterbi_encode(
        rotated, L=L, k=k, decode_fn=decode_fn
    )

    # Step 3: reconstruct the rotated walk's state sequence and read off
    # state[T/2]. Its top (L-k) bits are the overlap.
    mask = (1 << L) - 1
    s = int(start_rot)
    state_at_half = 0
    for t in range(T):
        s = ((s << k) | int(bs_rot[t])) & mask
        if t == T // 2:
            state_at_half = s
            break
    overlap_value = state_at_half >> k  # top (L-k) bits

    # Step 4: encode the original sequence with the overlap constraint
    bitstream, start_state, recon, mse = viterbi_encode_constrained(
        sequence, L=L, k=k, decode_fn=decode_fn,
        overlap_bits=overlap_bits, overlap_value=overlap_value,
    )

    return bitstream, start_state, recon, mse, overlap_value

# ============================================================================
# V-generic versions (V >= 1)
# ============================================================================

def precompute_predecessors_v(L, k, V):
    """For each state s in [0, 2^L), list its 2^(kV) predecessors under the
    bitshift rule next = ((s' << kV) | c) & mask."""
    n_states = 1 << L
    n_pred = 1 << (k * V)
    kV = k * V

    preds = np.zeros((n_states, n_pred), dtype=np.int32)
    for s in range(n_states):
        top = s >> kV  # s's top (L - kV) bits
        for p_top in range(n_pred):
            preds[s, p_top] = (p_top << (L - kV)) | top
    return preds


def precompute_codebook_v(L, V, decode_fn):
    """Materialize decode_fn(s) for every state. Returns (2^L, V).

    decode_fn should accept a uint32 numpy array and return either
    shape (N,) for V=1 or shape (N, V) for V>=1.
    """
    n_states = 1 << L
    states = np.arange(n_states, dtype=np.uint32)
    result = decode_fn(states)
    if result.ndim == 1:
        result = result[:, None]
    assert result.shape == (n_states, V), (
        f"decode_fn returned shape {result.shape}, expected ({n_states}, {V})"
    )
    return result.astype(np.float32)


def viterbi_encode_v(sequence, L, k, V, decode_fn, return_trace=False):
    """Generalized Viterbi encoder, V >= 1.

    sequence: float32, shape (T,) where T is a multiple of V
    Returns: (bitstream, start_state, recon, mse)
      bitstream: int32 shape (T/V,), each entry is kV bits
      start_state: int, the L-bit state before timestep 0
      recon: float32 shape (T,)
      mse: float
    """
    T = len(sequence)
    assert T % V == 0, f"Sequence length {T} must be divisible by V={V}"
    n_steps = T // V
    n_states = 1 << L
    kV = k * V

    codebook = precompute_codebook_v(L, V, decode_fn)     # (2^L, V)
    preds = precompute_predecessors_v(L, k, V)            # (2^L, 2^(kV))

    seq_steps = sequence.reshape(n_steps, V)              # (n_steps, V)

    cum_err = np.zeros(n_states, dtype=np.float32)
    backpointers = np.zeros((n_steps, n_states), dtype=np.int32)

    if return_trace:
        trace_err = np.zeros((n_steps, n_states), dtype=np.float32)

    for t in range(n_steps):
        w_vec = seq_steps[t]                              # (V,)
        diffs = codebook - w_vec[None, :]                 # (2^L, V)
        local_cost = (diffs * diffs).sum(axis=1)          # (2^L,)

        pred_costs = cum_err[preds]                        # (2^L, 2^(kV))
        new_cum = pred_costs.min(axis=1) + local_cost
        best_pred_idx = pred_costs.argmin(axis=1)
        backpointers[t] = preds[np.arange(n_states), best_pred_idx]
        cum_err = new_cum
        if return_trace:
            trace_err[t] = cum_err

    final_state = int(cum_err.argmin())

    states_walk = np.zeros(n_steps, dtype=np.int32)
    states_walk[n_steps - 1] = final_state
    for t in range(n_steps - 1, 0, -1):
        states_walk[t - 1] = backpointers[t, states_walk[t]]
    start_state = int(backpointers[0, states_walk[0]])

    mask_kV = (1 << kV) - 1
    bitstream = (states_walk & mask_kV).astype(np.int32)

    recon_steps = codebook[states_walk]                   # (n_steps, V)
    recon = recon_steps.reshape(-1)                       # (T,)
    mse = float(np.mean((recon - sequence) ** 2))

    if return_trace:
        return bitstream, start_state, recon, mse, trace_err, backpointers
    return bitstream, start_state, recon, mse


def viterbi_decode_v(bitstream, start_state, L, k, V, decode_fn):
    """Replay a V-aware walk from bitstream + start_state."""
    n_steps = len(bitstream)
    mask = (1 << L) - 1
    kV = k * V
    mask_kV = (1 << kV) - 1

    states = np.zeros(n_steps, dtype=np.uint32)
    s = np.uint32(start_state)
    for t in range(n_steps):
        s = ((s << np.uint32(kV)) | np.uint32(bitstream[t] & mask_kV)) & np.uint32(mask)
        states[t] = s

    codebook = precompute_codebook_v(L, V, decode_fn)
    recon_steps = codebook[states]
    return recon_steps.reshape(-1).astype(np.float32)


def viterbi_encode_constrained_v(sequence, L, k, V, decode_fn,
                                   overlap_bits, overlap_value):
    """Constrained V-generic Viterbi for tail-biting inner loop.

    s_0 top `overlap_bits` bits must equal overlap_value.
    s_{n_steps-1} bottom `overlap_bits` bits must equal overlap_value.
    """
    T = len(sequence)
    assert T % V == 0
    n_steps = T // V
    n_states = 1 << L
    kV = k * V

    codebook = precompute_codebook_v(L, V, decode_fn)
    preds = precompute_predecessors_v(L, k, V)
    seq_steps = sequence.reshape(n_steps, V)

    INF = np.float32(1e18)
    cum_err = np.zeros(n_states, dtype=np.float32)
    backpointers = np.zeros((n_steps, n_states), dtype=np.int32)

    top_mask = ((1 << overlap_bits) - 1) << (L - overlap_bits)
    target_top = (overlap_value & ((1 << overlap_bits) - 1)) << (L - overlap_bits)
    valid_s0 = (np.arange(n_states) & top_mask) == target_top

    for t in range(n_steps):
        w_vec = seq_steps[t]
        diffs = codebook - w_vec[None, :]
        local_cost = (diffs * diffs).sum(axis=1)
        pred_costs = cum_err[preds]
        new_cum = pred_costs.min(axis=1) + local_cost
        best_pred_idx = pred_costs.argmin(axis=1)
        backpointers[t] = preds[np.arange(n_states), best_pred_idx]

        if t == 0:
            new_cum = np.where(valid_s0, new_cum, INF)

        cum_err = new_cum

    bottom_mask = (1 << overlap_bits) - 1
    target_bottom = overlap_value & bottom_mask
    valid_ends = (np.arange(n_states) & bottom_mask) == target_bottom

    masked_err = np.where(valid_ends, cum_err, INF)
    final_state = int(masked_err.argmin())
    if float(masked_err[final_state]) >= INF / 2:
        raise RuntimeError(
            f"No valid tail-biting walk for overlap_value={overlap_value}"
        )

    states_walk = np.zeros(n_steps, dtype=np.int32)
    states_walk[n_steps - 1] = final_state
    for t in range(n_steps - 1, 0, -1):
        states_walk[t - 1] = backpointers[t, states_walk[t]]
    start_state = int(backpointers[0, states_walk[0]])

    mask_kV = (1 << kV) - 1
    bitstream = (states_walk & mask_kV).astype(np.int32)
    recon = codebook[states_walk].reshape(-1)
    mse = float(np.mean((recon - sequence) ** 2))

    return bitstream, start_state, recon, mse


def viterbi_encode_tailbiting_v(sequence, L, k, V, decode_fn):
    """Tail-biting via Algorithm 4, V-generic."""
    T = len(sequence)
    assert T % V == 0
    n_steps = T // V
    overlap_bits = L - k * V

    # Right-rotate by T/2 weights (equivalently n_steps/2 steps)
    # Keep T/2 divisible by V
    half_weights = (n_steps // 2) * V
    rotated = np.roll(sequence, half_weights)

    bs_rot, start_rot, _, _ = viterbi_encode_v(rotated, L, k, V, decode_fn)

    # Replay to find state at step n_steps/2; its top (L-kV) bits are the overlap
    mask = (1 << L) - 1
    kV = k * V
    s = int(start_rot)
    state_at_half = 0
    for t in range(n_steps):
        s = ((s << kV) | int(bs_rot[t])) & mask
        if t == n_steps // 2:
            state_at_half = s
            break
    overlap_value = state_at_half >> kV

    bitstream, start_state, recon, mse = viterbi_encode_constrained_v(
        sequence, L, k, V, decode_fn,
        overlap_bits=overlap_bits, overlap_value=overlap_value,
    )
    return bitstream, start_state, recon, mse, overlap_value


def viterbi_encode_v_batched(sequences, L, k, V, decode_fn):
    """Batched V-generic Viterbi encoder, fast version.

    Exploits two structural facts:
      1. Bitshift trellis: all states with the same top (L-kV) bits share
         predecessors. cum_err[preds[s, p]] is just cum_err viewed as
         (n_pred, n_top), so we take min over the small axis directly,
         skipping a 16x gather.
      2. Local cost via matmul: ||cb[s] - w[b]||^2 expands to
         ||cb[s]||^2 - 2*(w @ cb^T)[b,s] + ||w[b]||^2. The dot product is
         a BLAS matmul.
    """
    B, T = sequences.shape
    assert T % V == 0, f"T={T} must be divisible by V={V}"
    n_steps = T // V
    n_states = 1 << L
    kV = k * V
    n_pred = 1 << kV
    n_top = n_states >> kV  # = n_states / n_pred

    codebook = precompute_codebook_v(L, V, decode_fn)         # (n_states, V)
    cb_sq_norms = (codebook ** 2).sum(axis=1).astype(np.float32)  # (n_states,)

    seq_steps = sequences.reshape(B, n_steps, V).astype(np.float32)

    cum_err = np.zeros((B, n_states), dtype=np.float32)
    backpointers = np.zeros((n_steps, B, n_states), dtype=np.int32)

    # state_tops[s] = s >> kV — used to broadcast min_pred_cost back to all states
    state_tops = (np.arange(n_states, dtype=np.int32) >> kV)  # (n_states,)

    for t in range(n_steps):
        w_vec = seq_steps[:, t, :]                            # (B, V)

        # local_cost[b, s] = ||cb[s] - w[b]||^2 via expansion + matmul
        # = cb_sq_norms[s] - 2 * (w @ cb.T)[b, s] + (w[b]**2).sum()
        w_dot_cb = w_vec @ codebook.T                          # (B, n_states), BLAS
        w_sq = (w_vec ** 2).sum(axis=1, keepdims=True)         # (B, 1)
        local_cost = cb_sq_norms[None, :] - 2.0 * w_dot_cb + w_sq

        # Predecessor reduction without a gather:
        # cum_err viewed as (B, n_pred, n_top), min over n_pred axis
        cum_err_3d = cum_err.reshape(B, n_pred, n_top)
        min_pred_cost = cum_err_3d.min(axis=1)                 # (B, n_top)
        best_p = cum_err_3d.argmin(axis=1).astype(np.int32)    # (B, n_top)

        # Broadcast min_pred_cost back to all states via state_tops
        new_cum = min_pred_cost[:, state_tops] + local_cost    # (B, n_states)

        # Backpointer: preds[s, best_p[s>>kV]] = best_p * n_top + s_top
        backpointers[t] = (
            best_p[:, state_tops] * n_top + state_tops[None, :]
        )

        cum_err = new_cum

    final_states = cum_err.argmin(axis=1).astype(np.int32)

    states_walk = np.zeros((B, n_steps), dtype=np.int32)
    states_walk[:, n_steps - 1] = final_states
    for t in range(n_steps - 1, 0, -1):
        states_walk[:, t - 1] = backpointers[t][np.arange(B), states_walk[:, t]]
    start_states = backpointers[0][np.arange(B), states_walk[:, 0]].astype(np.int32)

    mask_kV = (1 << kV) - 1
    bitstreams = (states_walk & mask_kV).astype(np.int32)

    recon_steps = codebook[states_walk]                        # (B, n_steps, V)
    recons = recon_steps.reshape(B, T).astype(np.float32)
    mses = ((recons - sequences.astype(np.float32)) ** 2).mean(axis=1).astype(np.float32)

    return bitstreams, start_states, recons, mses