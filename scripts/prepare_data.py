"""
Download / prepare a training corpus.

By default this fetches "Tiny Shakespeare" (~1 MB) to data/input.txt. With
--bin it also tokenizes the text and writes memory-mapped train.bin / val.bin
(plus tokenizer.json) into --out_dir, so large corpora can be trained without
loading them fully into RAM.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --bin --tokenizer=bpe --vocab_size=1024
    python scripts/prepare_data.py --url=https://example.com/corpus.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import write_bin           # noqa: E402
from tokenizer import build_tokenizer   # noqa: E402

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=TINY_SHAKESPEARE_URL)
    parser.add_argument("--out", default="data/input.txt")
    parser.add_argument("--bin", action="store_true",
                        help="also write tokenized train.bin/val.bin")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--tokenizer", default="char", choices=["char", "bpe", "gpt2"])
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    try:
        import requests
        print(f"Downloading {args.url} ...")
        resp = requests.get(args.url, timeout=30)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:  # noqa: BLE001 - fall back to the bundled sample
        print(f"Download failed ({e}). Falling back to the bundled sample corpus.")
        with open(os.path.join("data", "sample.txt"), "r", encoding="utf-8") as f:
            text = f.read()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {len(text):,} characters to {args.out}")

    if args.bin:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Tokenizing with the {args.tokenizer} tokenizer ...")
        tok = build_tokenizer(args.tokenizer, text, args.vocab_size)
        tok.save(os.path.join(args.out_dir, "tokenizer.json"))
        ids = tok.encode(text)
        n_val = int(len(ids) * args.val_split)
        write_bin(ids[: len(ids) - n_val], os.path.join(args.out_dir, "train.bin"))
        write_bin(ids[len(ids) - n_val :], os.path.join(args.out_dir, "val.bin"))
        print(f"Wrote {len(ids):,} tokens to {args.out_dir}/train.bin + val.bin "
              f"(vocab size {tok.vocab_size}). Train with: python train.py")


if __name__ == "__main__":
    main()
