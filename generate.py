"""
Generate text from a trained checkpoint.

Uses the model's KV cache for fast autoregressive decoding, and can optionally
quantize the weights first for lower-RAM / faster inference.

Usage:
    python generate.py --prompt="To be, or not to be" --max_new_tokens=300
    python generate.py --quantize=int8                       # CPU, no extra deps
    python generate.py --quantize=int4                       # needs torchao
    python generate.py --quantize=nf4 --device=cuda          # needs bitsandbytes
    python generate.py --temperature=0.8 --top_k=40 --no_cache
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from quantize import model_size_mb, quantize_model
from tokenizer import load_tokenizer


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(out_dir: str, device: str, quantize: str = "none"):
    """Load a checkpoint + tokenizer, optionally quantizing the model."""
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    tok_path = os.path.join(out_dir, "tokenizer.json")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first with train.py.")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if quantize != "none":
        size_before = model_size_mb(model)
        model = quantize_model(model, quantize, device)
        print(f"Quantized to {quantize}: {size_before:.1f} MB -> "
              f"{model_size_mb(model):.1f} MB")
    else:
        model = model.to(device)

    return model, load_tokenizer(tok_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained small GPT.")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40, help="0 disables top-k")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantize", default="none",
                        choices=["none", "int8", "int4", "fp4", "nf4"])
    parser.add_argument("--no_cache", action="store_true", help="disable the KV cache")
    args = parser.parse_args()

    device = resolve_device(args.device)
    # int8 dynamic quantization runs on CPU.
    if args.quantize == "int8":
        device = "cpu"
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.out_dir, device, args.quantize)

    start_ids = tokenizer.encode(args.prompt) or tokenizer.encode("\n")
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    top_k = args.top_k if args.top_k > 0 else None
    for s in range(args.num_samples):
        out = model.generate(
            idx, args.max_new_tokens,
            temperature=args.temperature, top_k=top_k,
            use_cache=not args.no_cache,
        )
        print(f"\n----- sample {s + 1} -----")
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
