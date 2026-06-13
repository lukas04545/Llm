"""
Generate a training corpus with the DeepSeek API (teacher -> student distillation).

A small model can't be trained *through* a remote API (no gradients, and the
tokenizers differ), but a strong model like DeepSeek can act as a *teacher* that
writes high-quality, on-topic text. We save that text as a corpus and train the
small GPT on it with the normal pipeline -- "distillation by data".

Setup:
    export DEEPSEEK_API_KEY=sk-...          # get one at https://platform.deepseek.com
    python scripts/deepseek_gen.py --topic="bedtime stories for kids" --num_docs=200
    python train.py --data_path=data/deepseek.txt

Or tokenize straight to memmap .bin for the streaming/large-data path:
    python scripts/deepseek_gen.py --num_docs=500 --bin --tokenizer=bpe --vocab_size=1024
    python train.py

The DeepSeek API is OpenAI-compatible; this script uses plain `requests`, so no
extra dependency is needed.
"""

import argparse
import os
import random
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_BASE_URL = "https://api.deepseek.com"
DOC_SEPARATOR = "\n\n<|endoftext|>\n\n"  # clear boundary between generated docs


def call_deepseek(base_url, api_key, model, messages, temperature, max_tokens,
                  retries=4):
    """One chat-completion call with simple exponential-backoff retries."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            # 429 / 5xx -> retry; other 4xx -> fail fast with the server message.
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"DeepSeek API {resp.status_code}: {resp.text[:300]}")
            print(f"  retry {attempt + 1}/{retries} (HTTP {resp.status_code}) ...")
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}/{retries} ({e}) ...")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("DeepSeek API failed after retries.")


def build_messages(topic, prompt, nonce):
    """Construct the system+user messages for one diverse document."""
    system = (
        "You are a prolific writer generating clean, self-contained training text. "
        "Write natural prose only -- no markdown, headings, lists, or meta commentary. "
        "Each response must be a single, complete, standalone passage."
    )
    if prompt:
        user = f"{prompt}\n\n(Variation #{nonce}: make this one distinct from others.)"
    else:
        user = (
            f"Write a unique, self-contained passage (a few paragraphs) about: {topic}. "
            f"Vary the style, characters, and wording. Variation #{nonce}."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a corpus with DeepSeek.")
    parser.add_argument("--topic", default="short stories and everyday conversations",
                        help="theme for generated documents")
    parser.add_argument("--prompt", default="", help="explicit instruction (overrides --topic)")
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=800, help="per document")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default="deepseek-chat",
                        help="deepseek-chat (V3) or deepseek-reasoner (R1)")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", default="data/deepseek.txt")
    parser.add_argument("--append", action="store_true",
                        help="append to an existing corpus instead of overwriting")
    parser.add_argument("--seed", type=int, default=1337)
    # Optional: tokenize straight to memmap .bin for train.py auto-detection.
    parser.add_argument("--bin", action="store_true")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--tokenizer", default="char", choices=["char", "bpe", "gpt2"])
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--dry_run", action="store_true",
                        help="print the first request and exit (no API call)")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.dry_run:
        import json
        print(json.dumps(build_messages(args.topic, args.prompt, 1), indent=2))
        print(f"\n[dry run] would request {args.num_docs} docs from "
              f"{args.model} at {args.base_url}")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("Set DEEPSEEK_API_KEY in your environment (https://platform.deepseek.com).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    chars_total = 0
    with open(args.out, mode, encoding="utf-8") as f:
        for i in range(args.num_docs):
            nonce = random.randint(0, 1_000_000)
            messages = build_messages(args.topic, args.prompt, nonce)
            text = call_deepseek(args.base_url, api_key, args.model, messages,
                                 args.temperature, args.max_tokens).strip()
            f.write(text + DOC_SEPARATOR)
            f.flush()
            chars_total += len(text)
            print(f"[{i + 1}/{args.num_docs}] +{len(text)} chars "
                  f"(total {chars_total:,})")

    print(f"\nWrote ~{chars_total:,} characters to {args.out}")

    if args.bin:
        from dataset import write_bin
        from tokenizer import build_tokenizer
        with open(args.out, "r", encoding="utf-8") as f:
            text = f.read()
        os.makedirs(args.out_dir, exist_ok=True)
        tok = build_tokenizer(args.tokenizer, text, args.vocab_size)
        tok.save(os.path.join(args.out_dir, "tokenizer.json"))
        ids = tok.encode(text)
        n_val = int(len(ids) * args.val_split)
        write_bin(ids[: len(ids) - n_val], os.path.join(args.out_dir, "train.bin"))
        write_bin(ids[len(ids) - n_val :], os.path.join(args.out_dir, "val.bin"))
        print(f"Tokenized to {args.out_dir}/train.bin + val.bin "
              f"({len(ids):,} tokens). Train with: python train.py")
    else:
        print(f"Train with: python train.py --data_path={args.out}")


if __name__ == "__main__":
    main()
