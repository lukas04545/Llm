"""
Build a training corpus from arbitrary web pages.

Fetches a list of URLs, strips HTML to plain text, and appends everything into
a single text file you can train on (optionally writing memmap .bin files too).
This is the "train on text from the internet" path for specific sites; for
web-scale streaming use `train.py --stream=fineweb` instead.

Usage:
    python scripts/fetch_web.py --urls https://a.com https://b.org --out data/web.txt
    python scripts/fetch_web.py --url_file urls.txt --bin --tokenizer=bpe
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def html_to_text(html: str) -> str:
    """Very small HTML -> text cleaner (no external dependency)."""
    html = re.sub(r"(?is)<(script|style|head|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)          # drop tags
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)       # drop HTML entities
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="*", default=[])
    parser.add_argument("--url_file", default=None, help="text file, one URL per line")
    parser.add_argument("--out", default="data/web.txt")
    parser.add_argument("--bin", action="store_true")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--tokenizer", default="char", choices=["char", "bpe", "gpt2"])
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    urls = list(args.urls)
    if args.url_file:
        with open(args.url_file) as f:
            urls += [line.strip() for line in f if line.strip()]
    if not urls:
        parser.error("Provide --urls and/or --url_file")

    import requests
    parts = []
    for url in urls:
        try:
            print(f"Fetching {url} ...")
            resp = requests.get(url, timeout=30, headers={"User-Agent": "small-llm/1.0"})
            resp.raise_for_status()
            parts.append(html_to_text(resp.text))
        except Exception as e:  # noqa: BLE001
            print(f"  skipped ({e})")

    text = "\n\n".join(parts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {len(text):,} characters from {len(parts)} page(s) to {args.out}")

    if args.bin:
        from dataset import write_bin
        from tokenizer import build_tokenizer
        os.makedirs(args.out_dir, exist_ok=True)
        tok = build_tokenizer(args.tokenizer, text, args.vocab_size)
        tok.save(os.path.join(args.out_dir, "tokenizer.json"))
        ids = tok.encode(text)
        n_val = int(len(ids) * args.val_split)
        write_bin(ids[: len(ids) - n_val], os.path.join(args.out_dir, "train.bin"))
        write_bin(ids[len(ids) - n_val :], os.path.join(args.out_dir, "val.bin"))
        print(f"Wrote {len(ids):,} tokens to {args.out_dir}/train.bin + val.bin")


if __name__ == "__main__":
    main()
