"""Quick discovery: dump OlmoeExperts and OlmoeSparseMoeBlock internals.

Prints the parameter shapes inside the fused container plus the gate's
return format. Tells us what we need for the redesigned A.4 enumeration.
"""
import torch
from transformers import OlmoeForCausalLM, AutoTokenizer

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"


def main():
    print("Loading OLMoE in bf16...")
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    moe = model.model.layers[0].mlp
    print(f"\nMoE block class: {type(moe).__name__}")
    print(f"MoE block direct children:")
    for name, child in moe.named_children():
        print(f"  {name}: {type(child).__name__}")

    print(f"\n=== mlp.gate ===")
    print(f"  type: {type(moe.gate).__name__}")
    print(f"  parameters:")
    for n, p in moe.gate.named_parameters():
        print(f"    {n}: shape={tuple(p.shape)}, dtype={p.dtype}")

    # Try the gate's forward and inspect output
    print(f"\n  Running gate forward on a small input...")
    x = torch.randn(5, model.config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        out = moe.gate(x)
    print(f"  Output type: {type(out).__name__}")
    if isinstance(out, tuple):
        print(f"  Output is a tuple of length {len(out)}:")
        for i, item in enumerate(out):
            if isinstance(item, torch.Tensor):
                print(f"    [{i}]: tensor shape={tuple(item.shape)}, dtype={item.dtype}")
            else:
                print(f"    [{i}]: {type(item).__name__}")

    print(f"\n=== mlp.experts (OlmoeExperts) ===")
    experts = moe.experts
    print(f"  type: {type(experts).__name__}")
    print(f"  direct children: {list(experts.named_children())}")
    print(f"  parameters:")
    for n, p in experts.named_parameters():
        print(f"    {n}: shape={tuple(p.shape)}, dtype={p.dtype}")
    print(f"  buffers:")
    for n, b in experts.named_buffers():
        print(f"    {n}: shape={tuple(b.shape)}, dtype={b.dtype}")

    # Check if the experts module has a forward signature we can hook
    print(f"\n  experts.forward signature:")
    import inspect
    try:
        sig = inspect.signature(experts.forward)
        print(f"    {sig}")
    except (TypeError, ValueError) as e:
        print(f"    (could not introspect: {e})")

    # Print the first few lines of the forward source if accessible
    try:
        src = inspect.getsource(experts.forward)
        print(f"\n  experts.forward source (first 30 lines):")
        for line in src.split("\n")[:30]:
            print(f"    {line}")
    except (TypeError, OSError) as e:
        print(f"  (source not accessible: {e})")


if __name__ == "__main__":
    main()