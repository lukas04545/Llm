"""
Generate text from a trained checkpoint.

Usage:
    python generate.py --prompt="To be, or not to be" --max_new_tokens=300
    python generate.py --out_dir=out --temperature=0.8 --top_k=40
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained small GPT.")
    parser.add_argument("--out_dir", default="out", help="dir with ckpt.pt + tokenizer.json")
    parser.add_argument("--prompt", default="\n", help="text to condition on")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="lower = more greedy, higher = more random")
    parser.add_argument("--top_k", type=int, default=40,
                        help="only sample from the top-k tokens (0 disables)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first with train.py.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    tokenizer = CharTokenizer.load(tok_path)

    start_ids = tokenizer.encode(args.prompt) or tokenizer.encode("\n")
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    top_k = args.top_k if args.top_k > 0 else None
    for s in range(args.num_samples):
        out = model.generate(
            idx, args.max_new_tokens,
            temperature=args.temperature, top_k=top_k,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"\n----- sample {s + 1} -----")
        print(text)


if __name__ == "__main__":
    main()
