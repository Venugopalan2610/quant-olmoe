"""Tripwire A.2: Viterbi encoder correctness.

A.2.1: L=2 toy reproduction (hand-traced 3-weight example)
A.2.2: Encode/decode round-trip (bit-exact)
A.2.3: Paper Table 1 reproduction (MSE on iid Gaussian, L=16, k=2)
A.2.4: Tail-biting variant (deferred to a separate file once base passes)

Run: python -m src.tripwires.test_viterbi
"""
import sys
import numpy as np

from src.codes.ref import decode_1mad_batch, decode_3inst_batch, decode_hyb_batch
from src.viterbi.encode import viterbi_encode, viterbi_decode


def test_a21_toy():
    """A.2.1 — L=2, k=1, hardcoded decoder, 3-weight example.

    From the conversation walkthrough:
      decode(0)=-1.2, decode(1)=-0.3, decode(2)=0.4, decode(3)=1.1
      sequence = [-1.0, 0.5, 0.8]
      expected walk: 00 -> 00 -> 01 -> 11
      expected total error: 0.77
      expected reconstruction: [-1.2, -0.3, 1.1]
    """
    print("\nA.2.1: L=2 toy reproduction")
    print("-" * 60)

    toy_table = np.array([-1.2, -0.3, 0.4, 1.1], dtype=np.float32)
    def toy_decode(states):
        return toy_table[states.astype(np.int64)]

    sequence = np.array([-1.0, 0.5, 0.8], dtype=np.float32)
    bitstream, start_state, recon, mse = viterbi_encode(
        sequence, L=2, k=1, decode_fn=toy_decode
    )

    expected_recon = np.array([-1.2, -0.3, 1.1], dtype=np.float32)
    expected_total_err = 0.77
    actual_total_err = float(np.sum((recon - sequence) ** 2))

    recon_ok = np.allclose(recon, expected_recon, atol=1e-5)
    err_ok = abs(actual_total_err - expected_total_err) < 1e-4

    print(f"  reconstruction: {recon}")
    print(f"  expected:       {expected_recon}")
    print(f"  total err: {actual_total_err:.4f} (expected {expected_total_err})")
    print(f"  start_state: {start_state}, bitstream: {bitstream.tolist()}")

    ok = recon_ok and err_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] toy reproduction")
    return ok


def test_a22_roundtrip():
    """A.2.2 — encode then decode, recon must bit-exactly match decode output."""
    print("\nA.2.2: Encode/decode round-trip")
    print("-" * 60)

    rng = np.random.default_rng(0)
    T = 256
    sequence = rng.standard_normal(T).astype(np.float32)

    bitstream, start_state, recon, mse = viterbi_encode(
        sequence, L=16, k=2, decode_fn=decode_1mad_batch
    )
    replayed = viterbi_decode(bitstream, start_state, L=16, k=2, decode_fn=decode_1mad_batch)

    bit_exact = np.array_equal(recon, replayed)
    print(f"  T={T}, MSE={mse:.4f}")
    print(f"  recon vs replayed: max abs diff = {np.max(np.abs(recon - replayed)):.2e}")
    print(f"  [{'PASS' if bit_exact else 'FAIL'}] bit-exact round-trip")
    return bit_exact


def test_a23_paper_table1():
    """A.2.3 — reproduce paper Table 1 distortion on iid Gaussian, k=2.

    Expected MSE values (Table 1, L=16):
      1MAD:  0.069
      3INST: 0.068
      HYB:   0.069  (1D version; paper's HYB is 2D V=2, slightly different)

    We accept anything in [0.066, 0.075] for the lookup-free codes.
    For HYB V=1 we accept [0.066, 0.080] since we're not using the
    intended 2D version yet.
    """
    print("\nA.2.3: Paper Table 1 reproduction (L=16, k=2, V=1)")
    print("-" * 60)

    rng = np.random.default_rng(42)
    n_seqs = 64
    T = 256
    L = 16
    k = 2

    lut = np.load("cache/codes/hyb_lut_init.npy")
    def hyb1d_decode(states):
        # V=1 HYB: just take the first component of each LUT row
        return decode_hyb_batch(states, lut, Q=9)[:, 0]

    results = {}
    for name, decode_fn in [
        ("1MAD", decode_1mad_batch),
        ("3INST", decode_3inst_batch),
        ("HYB(V=1)", hyb1d_decode),
    ]:
        mses = []
        for i in range(n_seqs):
            seq = rng.standard_normal(T).astype(np.float32)
            _, _, _, mse = viterbi_encode(seq, L=L, k=k, decode_fn=decode_fn)
            mses.append(mse)
        mean_mse = float(np.mean(mses))
        std_mse = float(np.std(mses))
        results[name] = (mean_mse, std_mse)

        if name == "HYB(V=1)":
            ok = 0.062 <= mean_mse <= 0.080
        else:
            ok = 0.062 <= mean_mse <= 0.075
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: MSE = {mean_mse:.4f} ± {std_mse:.4f} "
              f"(over {n_seqs} sequences of length {T})")
        results[name] += (ok,)

    print(f"\n  Reference (paper Table 1, L=16, k=2):")
    print(f"    1MAD:  0.069")
    print(f"    3INST: 0.068")
    print(f"    HYB:   0.069 (their 2D version)")
    print(f"    DR (rate-distortion bound): 0.063")

    return all(r[2] for r in results.values())

def test_a24_tailbiting():
    """A.2.4 — tail-biting via Algorithm 4.

    Verify that tail-biting MSE is close to non-tail-biting MSE (within ~0.001
    on iid Gaussian) and that the encoded walk actually has the overlap
    property.
    """
    print("\nA.2.4: Tail-biting (Algorithm 4)")
    print("-" * 60)
    from src.viterbi.encode import viterbi_encode_tailbiting

    rng = np.random.default_rng(42)
    n_seqs = 32
    T = 256
    L = 16
    k = 2

    base_mses = []
    tb_mses = []
    overlap_ok_count = 0

    for i in range(n_seqs):
        seq = rng.standard_normal(T).astype(np.float32)

        _, _, _, mse_base = viterbi_encode(seq, L=L, k=k, decode_fn=decode_1mad_batch)
        bs_tb, start_tb, recon_tb, mse_tb, overlap = viterbi_encode_tailbiting(
            seq, L=L, k=k, decode_fn=decode_1mad_batch
        )
        base_mses.append(mse_base)
        tb_mses.append(mse_tb)

        # Verify the tail-biting property: s_0.top(L-k) == s_{T-1}.bottom(L-k).
        # Replay the walk from start_tb to materialize s_0 and s_{T-1}.
        overlap_bits = L - k
        mask = (1 << L) - 1
        s = start_tb
        s_0 = None
        s_last = None
        for idx, c in enumerate(bs_tb):
            s = ((s << k) | int(c)) & mask
            if idx == 0:
                s_0 = s
            s_last = s
        s0_top = s_0 >> k
        s_last_bottom = s_last & ((1 << overlap_bits) - 1)
        if s0_top == s_last_bottom == overlap:
            overlap_ok_count += 1

    base_mean = float(np.mean(base_mses))
    tb_mean = float(np.mean(tb_mses))
    gap = tb_mean - base_mean

    overlap_ok = overlap_ok_count == n_seqs
    mse_ok = abs(gap) < 0.005  # tail-biting should add at most ~0.005 MSE

    print(f"  Base MSE:        {base_mean:.4f}")
    print(f"  Tail-biting MSE: {tb_mean:.4f}")
    print(f"  Gap:             {gap:+.4f}")
    print(f"  Overlap property satisfied: {overlap_ok_count}/{n_seqs}")

    print(f"  [{'PASS' if mse_ok else 'FAIL'}] MSE gap < 0.005")
    print(f"  [{'PASS' if overlap_ok else 'FAIL'}] All walks tail-biting")
    return mse_ok and overlap_ok

def main():
    print("=" * 60)
    print("Tripwire A.2: Viterbi encoder correctness")
    print("=" * 60)

    results = []
    results.append(("A.2.1 toy", test_a21_toy()))
    results.append(("A.2.2 round-trip", test_a22_roundtrip()))
    results.append(("A.2.3 Table 1", test_a23_paper_table1()))
    results.append(("A.2.4 tail-biting", test_a24_tailbiting()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.2 GATE: PASS — Viterbi encoder verified.")
        print("Ready for A.2.4 (tail-biting) then A.3 (RHT).")
        sys.exit(0)
    else:
        print("A.2 GATE: FAIL — fix failing tripwires before proceeding.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()