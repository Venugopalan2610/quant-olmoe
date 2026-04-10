"""Tripwire A.4: OLMoE model adapter (fused experts version).

Run: python -m src.tripwires.test_olmoe
"""
import sys
import torch
from transformers import OlmoeForCausalLM, AutoTokenizer

from src.models.olmoe_adapter import (
    get_arch_config,
    discover_layer_structure,
    enumerate_quant_targets,
)


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"


def load_model_bf16():
    print("  Loading OLMoE in bf16...")
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    return model


def test_a41_enumeration(model):
    print("\nA.4.1: Module enumeration (fused storage)")
    print("-" * 60)

    cfg = get_arch_config(model)
    print(f"  Config: {cfg}")
    discover_layer_structure(model, layer_idx=0)

    targets = list(enumerate_quant_targets(model))
    n_attn = sum(1 for t in targets if t.kind == "attention")
    n_expert = sum(1 for t in targets if t.kind == "expert")

    expected_attn = cfg["num_hidden_layers"] * 4
    expected_expert = cfg["num_hidden_layers"] * cfg["num_experts"] * 3
    expected_total = expected_attn + expected_expert

    print(f"\n  Attention targets: {n_attn} (expected {expected_attn})")
    print(f"  Expert targets:    {n_expert} (expected {expected_expert})")
    print(f"  Total:             {len(targets)} (expected {expected_total})")

    # Verify uniqueness of names
    names = [t.name for t in targets]
    unique_ok = len(names) == len(set(names))
    print(f"  All names unique:  {unique_ok}")

    # Verify get/set roundtrip on a few random targets
    import random
    random.seed(0)
    sample = random.sample(targets, 5)
    roundtrip_ok = True
    for t in sample:
        original = t.get_weight()
        modified = original + 0.01
        t.set_weight(modified)
        readback = t.get_weight()
        if not torch.allclose(readback, modified, atol=1e-2):
            print(f"  ROUNDTRIP FAIL: {t.name}")
            roundtrip_ok = False
        # Restore
        t.set_weight(original)

    print(f"  get/set roundtrip on 5 samples: {'OK' if roundtrip_ok else 'FAIL'}")

    counts_ok = (n_attn == expected_attn and n_expert == expected_expert)
    ok = counts_ok and unique_ok and roundtrip_ok
    print(f"\n  [{'PASS' if ok else 'FAIL'}] enumeration")
    return ok


def test_a42_forward_sanity(model):
    print("\nA.4.2: Forward pass sanity")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt = "Bitcoin is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)
    last_logits = out.logits[0, -1, :]

    finite_ok = torch.isfinite(last_logits).all().item()
    scale = float(last_logits.float().std())
    scale_ok = 1.0 < scale < 50.0

    top_id = int(last_logits.argmax())
    top_token = tokenizer.decode([top_id])

    print(f"  Logits std: {scale:.3f}")
    print(f"  Top-1 next token: {top_token!r}")

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    text = tokenizer.decode(generated[0])
    print(f"  Generated: {text!r}")

    after = text[len(prompt):].strip()
    has_words = len([w for w in after.split() if w.isalpha()]) >= 3

    ok = finite_ok and scale_ok and has_words
    print(f"  [{'PASS' if ok else 'FAIL'}] forward sanity")
    return ok


def test_a43_router_dispatch(model):
    """A.4.3 — verify the router returns valid (top_k_weights, top_k_indices)."""
    print("\nA.4.3: Router dispatch pattern")
    print("-" * 60)

    cfg = get_arch_config(model)
    top_k = cfg["num_experts_per_tok"]
    num_experts = cfg["num_experts"]

    moe_block = model.model.layers[0].mlp
    captured = {}

    def gate_hook(module, inputs, output):
        # OlmoeTopKRouter returns (router_logits, top_k_weights, top_k_indices)
        captured["router_logits"] = output[0].detach()
        captured["top_k_weights"] = output[1].detach()
        captured["top_k_indices"] = output[2].detach()

    handle = moe_block.gate.register_forward_hook(gate_hook)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    rl = captured["router_logits"]
    tw = captured["top_k_weights"]
    ti = captured["top_k_indices"]

    print(f"  Router logits shape: {tuple(rl.shape)}")
    print(f"  top_k_weights shape: {tuple(tw.shape)}")
    print(f"  top_k_indices shape: {tuple(ti.shape)}")
    print(f"  Tokens in prompt: {n_tokens}")

    # Reshape to (flat_tokens, ...) regardless of whether HF flattened
    if ti.dim() == 3:
        ti = ti.reshape(-1, top_k)
        tw = tw.reshape(-1, top_k)
    flat_tokens = ti.shape[0]

    # Check shape consistency
    shape_ok = (
        ti.shape == (flat_tokens, top_k)
        and tw.shape == (flat_tokens, top_k)
        and rl.shape[-1] == num_experts
    )

    # Each token should have top_k unique experts
    unique_per_row = [len(set(ti[i].tolist())) for i in range(min(flat_tokens, 16))]
    all_unique = all(u == top_k for u in unique_per_row)

    # Indices in valid range
    valid_range = bool(((ti >= 0) & (ti < num_experts)).all().item())

    # Weights are nonneg and roughly sum to ~1 per token (renormalized)
    weights_ok = bool((tw >= 0).all().item())
    row_sums = tw.float().sum(dim=-1)
    # OLMoE does NOT renormalize after top-k (unlike Mixtral). Raw softmax
    # over 64 truncated to top-8 sums to ~0.4-0.5. Just check it's in a
    # plausible range, not == 1.
    sum_ok = bool(((row_sums > 0.2) & (row_sums < 0.8)).all().item())

    print(f"  First 3 tokens, top-{top_k} experts:")
    for i in range(min(3, flat_tokens)):
        print(f"    {ti[i].tolist()}  weights {[f'{w:.3f}' for w in tw[i].tolist()]}")
    print(f"  Row sums (raw softmax top-8, NOT renormalized): "
          f"mean={float(row_sums.mean()):.4f}, "
          f"min={float(row_sums.min()):.4f}, max={float(row_sums.max()):.4f}")

    # Per-expert load
    per_expert = torch.zeros(num_experts, dtype=torch.long)
    for row in ti:
        for e in row:
            per_expert[int(e)] += 1
    n_active = int((per_expert > 0).sum())
    print(f"  Active experts in this prompt: {n_active}/{num_experts}")

    ok = shape_ok and all_unique and valid_range and weights_ok and sum_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] dispatch pattern")
    return ok


def main():
    print("=" * 60)
    print("Tripwire A.4: OLMoE model adapter")
    print("=" * 60)

    model = load_model_bf16()

    results = []
    results.append(("A.4.1 enumeration", test_a41_enumeration(model)))
    results.append(("A.4.2 forward sanity", test_a42_forward_sanity(model)))
    results.append(("A.4.3 router dispatch", test_a43_router_dispatch(model)))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("A.4 GATE: PASS — OLMoE adapter verified.")
        print("Ready for A.5 (Hessian collection).")
        sys.exit(0)
    else:
        print("A.4 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()