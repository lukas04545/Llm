"""
Post-training quantization for fast, low-RAM inference.

Quantization is an *inference-time* technique: you train the model in
bf16/fp16, then compress the trained weights so generation needs less memory
and (on CPU) runs faster. A ladder of options is provided, from zero-dependency
int8 up to 4-bit:

    none  -- full precision (baseline)
    int8  -- dynamic int8 over Linear layers. No extra dependency. CPU. (default lite)
    int4  -- 4-bit weight-only via torchao. CPU + CUDA.        pip install torchao
    fp4   -- 4-bit float (FP4) via bitsandbytes. CUDA / Colab. pip install bitsandbytes
    nf4   -- 4-bit NormalFloat via bitsandbytes. CUDA / Colab. pip install bitsandbytes

Usage as a script (loads a checkpoint, quantizes, reports size, re-saves):
    python quantize.py --out_dir=out --quantize=int8
    python quantize.py --quantize=int4
"""

import argparse
import os

import torch
import torch.nn as nn

from model import GPT, GPTConfig


def quantize_model(model: nn.Module, mode: str, device: str = "cpu") -> nn.Module:
    """Return a quantized copy/version of `model` for the given mode."""
    mode = mode.lower()
    if mode in ("none", ""):
        return model
    if mode == "int8":
        return _quant_int8(model)
    if mode == "int4":
        return _quant_int4_torchao(model, device)
    if mode in ("fp4", "nf4"):
        return _quant_4bit_bnb(model, mode, device)
    raise ValueError(f"Unknown quantization mode: {mode}")


def _quant_int8(model: nn.Module) -> nn.Module:
    """Dynamic int8 quantization of all Linear layers (CPU). No extra deps."""
    model = model.to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def _quant_int4_torchao(model: nn.Module, device: str) -> nn.Module:
    try:
        from torchao.quantization import int4_weight_only, quantize_
    except ImportError as e:
        raise ImportError(
            "int4 quantization needs torchao. Install with: pip install torchao"
        ) from e
    model = model.to(device).eval()
    quantize_(model, int4_weight_only())
    return model


def _quant_4bit_bnb(model: nn.Module, mode: str, device: str) -> nn.Module:
    """Swap nn.Linear for bitsandbytes 4-bit Linear (FP4 or NF4). CUDA."""
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            f"{mode} quantization needs bitsandbytes. Install with: pip install bitsandbytes"
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(f"{mode} (bitsandbytes) requires a CUDA GPU.")

    quant_type = "fp4" if mode == "fp4" else "nf4"
    model = model.to("cuda").eval()
    _swap_linear_4bit(model, bnb, quant_type)
    return model.to("cuda")


def _swap_linear_4bit(module: nn.Module, bnb, quant_type: str) -> None:
    """Recursively replace nn.Linear with bnb.nn.Linear4bit, copying weights."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new = bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16, quant_type=quant_type,
            )
            new.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type=quant_type
            )
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, new)
        else:
            _swap_linear_4bit(child, bnb, quant_type)


def model_size_mb(model: nn.Module) -> float:
    """Size of a model's serialized weights, in MB.

    Serializing the state dict (rather than summing `.parameters()`) correctly
    accounts for packed int8 / 4-bit storage produced by dynamic quantization,
    torchao, and bitsandbytes, which don't expose weights as plain parameters.
    """
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a trained checkpoint.")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--quantize", default="int8",
                        choices=["none", "int8", "int4", "fp4", "nf4"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    before = model_size_mb(model)
    qmodel = quantize_model(model, args.quantize, args.device)
    after = model_size_mb(qmodel)

    print(f"Quantization: {args.quantize}")
    print(f"  size before: {before:.2f} MB")
    print(f"  size after:  {after:.2f} MB  ({before / max(after, 1e-9):.2f}x smaller)")


if __name__ == "__main__":
    main()
