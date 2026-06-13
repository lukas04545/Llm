"""
Download a training corpus.

By default this fetches the "Tiny Shakespeare" dataset (~1 MB) which trains
quickly into something that produces Shakespeare-like text. You can point the
training script at any UTF-8 text file instead via --data_path.

Usage:
    python scripts/prepare_data.py                 # -> data/input.txt
    python scripts/prepare_data.py --out data/my.txt
"""

import argparse
import os

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=TINY_SHAKESPEARE_URL)
    parser.add_argument("--out", default="data/input.txt")
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


if __name__ == "__main__":
    main()
