"""
Train a byte-level BPE tokenizer from a text file and save it.

Usage:
    python scripts/train_bpe.py --data_path=data/input.txt --vocab_size=1024
    python scripts/train_bpe.py --out=out/tokenizer.json --vocab_size=4096
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/input.txt")
    parser.add_argument("--out", default="out/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=1024)
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Training BPE (vocab_size={args.vocab_size}) on {len(text):,} chars ...")
    tok = BPETokenizer.train(text, args.vocab_size, verbose=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)

    sample = text[:200]
    ids = tok.encode(sample)
    print(f"Saved to {args.out} | vocab size {tok.vocab_size}")
    print(f"Compression on a 200-char sample: {len(sample)} chars -> {len(ids)} tokens "
          f"({len(sample) / max(1, len(ids)):.2f} chars/token)")


if __name__ == "__main__":
    main()
