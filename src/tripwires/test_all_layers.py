"""Tripwire A.5.4: all-layer Hessian collection.

A.5.4.1: All 16 L{NN} directories exist with 132 Hessian files each
A.5.4.2: Token counts per layer sum to expected (16,777,216 for experts,
         2,097,152 for attention)
A.5.4.3: No NaN/Inf in any sampled Hessian
A.5.4.4: Expert load heatmap shows specialization pattern
A.5.4.5: End-to-end forward: final norm + LM head on layer_16_input
         produces finite logits and plausible tokens

Run: python -m src.tripwires.test_all_layers
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import OlmoeForCausalLM, AutoTokenizer

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
HIDDEN_DIR = "cache/hidden_states"
HESSIAN_DIR = "cache/hessians"
PLOTS_DIR = "plots/hessian_collection"


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Tripwire A.5.4: all-layer Hessian collection")
    print("=" * 60)

    num_layers = 16
    num_experts = 64
    top_k = 8
    n_tokens = 2048 * 1024
    expected_expert_sum = n_tokens * top_k
    expected_attn_per_proj = n_tokens

    # A.5.4.1 — all layer dirs populated
    print("\nA.5.4.1: file counts per layer")
    print("-" * 60)
    files_ok = True
    for L in range(num_layers):
        d = os.path.join(HESSIAN_DIR, f"L{L:02d}")
        if not os.path.exists(d):
            print(f"  L{L:02d}: MISSING")
            files_ok = False
            continue
        files = os.listdir(d)
        attn = sorted(f for f in files if f.startswith("attn_"))
        gu = sorted(f for f in files if f.endswith("_gate_up.pt"))
        dn = sorted(f for f in files if f.endswith("_down.pt"))
        ok = len(attn) == 4 and len(gu) == 64 and len(dn) == 64
        marker = "OK  " if ok else "FAIL"
        print(f"  L{L:02d} [{marker}]: {len(attn)} attn, {len(gu)} gate_up, {len(dn)} down")
        if not ok:
            files_ok = False
    print(f"  [{'PASS' if files_ok else 'FAIL'}] all layer dirs have 4+64+64 files")

    # A.5.4.2 — token counts
    print("\nA.5.4.2: token counts per layer")
    print("-" * 60)
    counts_ok = True
    for L in range(num_layers):
        d = os.path.join(HESSIAN_DIR, f"L{L:02d}")
        # Attention: each of 4 projections sees n_tokens
        attn_sum = 0
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            data = torch.load(os.path.join(d, f"attn_{proj}.pt"), weights_only=True)
            attn_sum += data["n_tokens"]
        # Expert gate_up: sum over 64 experts should equal n_tokens * top_k
        gu_sum = 0
        dn_sum = 0
        for e in range(num_experts):
            gu = torch.load(os.path.join(d, f"expert_{e:02d}_gate_up.pt"),
                            weights_only=True)
            dn = torch.load(os.path.join(d, f"expert_{e:02d}_down.pt"),
                            weights_only=True)
            gu_sum += gu["n_tokens"]
            dn_sum += dn["n_tokens"]
        attn_ok = attn_sum == 4 * expected_attn_per_proj
        gu_ok = gu_sum == expected_expert_sum
        dn_ok = dn_sum == expected_expert_sum
        row_ok = attn_ok and gu_ok and dn_ok
        marker = "OK  " if row_ok else "FAIL"
        print(f"  L{L:02d} [{marker}]: attn={attn_sum:,} "
              f"gu={gu_sum:,} dn={dn_sum:,}")
        if not row_ok:
            counts_ok = False
    print(f"  [{'PASS' if counts_ok else 'FAIL'}] all token counts correct")

    # A.5.4.3 — NaN/Inf sweep
    print("\nA.5.4.3: NaN/Inf sweep on 12 random Hessians")
    print("-" * 60)
    rng = np.random.default_rng(0)
    finite_ok = True
    for _ in range(12):
        L = int(rng.integers(0, num_layers))
        e = int(rng.integers(0, num_experts))
        kind = "gate_up" if rng.random() < 0.5 else "down"
        path = os.path.join(HESSIAN_DIR, f"L{L:02d}", f"expert_{e:02d}_{kind}.pt")
        H = torch.load(path, weights_only=True)["H"]
        n_nan = int(torch.isnan(H).sum())
        n_inf = int(torch.isinf(H).sum())
        if n_nan or n_inf:
            print(f"  L{L:02d} E{e:02d} {kind}: NaN={n_nan}, Inf={n_inf} [FAIL]")
            finite_ok = False
    if finite_ok:
        print(f"  All 12 sampled Hessians finite")
    print(f"  [{'PASS' if finite_ok else 'FAIL'}] no NaN/Inf")

    # A.5.4.4 — expert load heatmap
    print("\nA.5.4.4: expert load heatmap")
    print("-" * 60)
    heatmap = np.zeros((num_layers, num_experts), dtype=np.int64)
    for L in range(num_layers):
        for e in range(num_experts):
            gu = torch.load(
                os.path.join(HESSIAN_DIR, f"L{L:02d}", f"expert_{e:02d}_gate_up.pt"),
                weights_only=True,
            )
            heatmap[L, e] = gu["n_tokens"]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xlabel("expert index")
    ax.set_ylabel("layer")
    ax.set_title(f"Per-expert token count ({num_layers} layers × {num_experts} experts)\n"
                 f"uniform rate: {expected_expert_sum // num_experts:,}")
    plt.colorbar(im, ax=ax, label="tokens routed")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "expert_load_heatmap.png"), dpi=100)
    plt.close()
    print(f"  Saved heatmap to {PLOTS_DIR}/expert_load_heatmap.png")

    uniform_rate = expected_expert_sum // num_experts
    min_load = int(heatmap.min())
    max_load = int(heatmap.max())
    print(f"  Global load range: min={min_load:,} max={max_load:,} "
          f"(uniform would be {uniform_rate:,})")
    load_ok = min_load > 10_000  # at least 10K tokens per expert per layer
    print(f"  [{'PASS' if load_ok else 'FAIL'}] all experts got >10K tokens")

    # A.5.4.5 — end-to-end forward verification
    print("\nA.5.4.5: end-to-end forward verification")
    print("-" * 60)
    final_in = os.path.join(HIDDEN_DIR, "layer_16_input")
    if not os.path.exists(final_in):
        print(f"  FAIL: {final_in} missing")
        sys.exit(1)

    print("  Loading model (for final norm + LM head)...")
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    final_norm = model.model.norm
    lm_head = model.lm_head

    shard = torch.load(
        os.path.join(final_in, "shard_0000.pt"), weights_only=True,
    )  # (B, seq, hidden) bf16

    with torch.no_grad():
        normed = final_norm(shard[:2])  # first 2 sequences for speed
        logits = lm_head(normed)  # (2, seq, vocab)

    finite_ok_e2e = bool(torch.isfinite(logits).all())
    std = float(logits.float().std())
    scale_ok_e2e = 1.0 < std < 50.0
    print(f"  Logits shape: {tuple(logits.shape)}")
    print(f"  Logits std: {std:.3f}")
    print(f"  All finite: {finite_ok_e2e}")

    # Decode top-1 for a few positions and check they form words
    top_ids = logits.argmax(dim=-1)[0, :20].tolist()
    decoded = tokenizer.decode(top_ids)
    print(f"  Top-1 decoded from first 20 positions of seq 0:")
    print(f"    {decoded!r}")

    # Weak sanity: at least some alphabetic tokens
    has_text = any(c.isalpha() for c in decoded)

    ok_e2e = finite_ok_e2e and scale_ok_e2e and has_text
    print(f"  [{'PASS' if ok_e2e else 'FAIL'}] final logits finite and sensible")

    print("\n" + "=" * 60)
    all_ok = files_ok and counts_ok and finite_ok and load_ok and ok_e2e
    if all_ok:
        print("A.5.4 GATE: PASS — all-layer Hessian collection verified.")
        print(f"Hessians at {HESSIAN_DIR}/L00..L15/")
        print("Ready for A.6 (calibration diagnostics).")
        sys.exit(0)
    else:
        print("A.5.4 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()