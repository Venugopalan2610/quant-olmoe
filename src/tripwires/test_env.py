"""Tripwire A.0.1: environment sanity.

Run: python -m src.tripwires.test_env
"""
import sys


def main():
    checks = []

    # Python version
    ok = sys.version_info >= (3, 11)
    checks.append(("Python >= 3.11", ok, f"{sys.version_info.major}.{sys.version_info.minor}"))

    # Torch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
        checks.append(("torch + CUDA", cuda_ok, f"{torch.__version__} / {device_name}"))
    except Exception as e:
        checks.append(("torch import", False, str(e)))

    # Transformers with OLMoE support
    try:
        from transformers import OlmoeForCausalLM
        import transformers
        checks.append(("transformers + OlmoeForCausalLM", True, transformers.__version__))
    except Exception as e:
        checks.append(("transformers import", False, str(e)))

    # QTIP submodule importable
    try:
        sys.path.insert(0, "external/qtip")
        import lib.codebook.bitshift  # noqa: F401
        checks.append(("qtip submodule", True, "importable"))
    except Exception as e:
        checks.append(("qtip submodule", False, str(e)))

    # HF Hub reachable
    try:
        from huggingface_hub import HfApi
        HfApi().model_info("allenai/OLMoE-1B-7B-0125")
        checks.append(("HF Hub reachable", True, "allenai/OLMoE-1B-7B-0125 visible"))
    except Exception as e:
        checks.append(("HF Hub", False, str(e)))

    print()
    print(f"{'CHECK':<40} {'STATUS':<8} DETAIL")
    print("-" * 80)
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{name:<40} {status:<8} {detail}")
        all_ok = all_ok and ok
    print()
    if all_ok:
        print("All environment checks passed. Ready for A.1.")
        sys.exit(0)
    else:
        print("Environment incomplete. Fix failing checks before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
