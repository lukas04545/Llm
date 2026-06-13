"""
Import an open-weight Llama-family model into this repo's format.

Saves a checkpoint (out/ckpt.pt) and the matching tokenizer (out/tokenizer.json)
so you can immediately generate, fine-tune (train.py --init_from=out/ckpt.pt), or
grow it (scripts/grow.py).

    pip install transformers
    python scripts/import_hf.py --repo=HuggingFaceTB/SmolLM-135M
    python generate.py --prompt="The meaning of life is"
    python train.py --init_from=out/ckpt.pt --data_path=data/mydata.txt   # fine-tune

Works with bias-free Llama-arch models (SmolLM, TinyLlama). Larger models need a
GPU and more RAM; some (e.g. gated Llama) require `huggingface-cli login`.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained import load_pretrained, model_args_from_config  # noqa: E402
from tokenizer import HFTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M",
                        help="Hugging Face model id (Llama-arch, bias-free)")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.repo} ...")
    model, cfg = load_pretrained(args.repo, device=args.device, dtype=dtype)
    print(f"Imported: {model.num_params() / 1e6:.1f}M params | {cfg.n_layer} layers "
          f"| n_embd {cfg.n_embd} | vocab {cfg.vocab_size}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_args": model_args_from_config(cfg),
        "iter": 0,
        "source": args.repo,
    }, os.path.join(args.out_dir, "ckpt.pt"))
    HFTokenizer.save_config(os.path.join(args.out_dir, "tokenizer.json"), args.repo)
    print(f"Saved to {args.out_dir}/ckpt.pt + tokenizer.json")
    print("Next: python generate.py --prompt='...'  |  python train.py --init_from=out/ckpt.pt ...")


if __name__ == "__main__":
    main()
