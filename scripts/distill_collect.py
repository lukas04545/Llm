"""
Collect a top-k logprob distillation dataset from the DeepSeek API.

This is the data-collection half of cross-model distillation. For each generated
document we ask the API for `logprobs` with `top_logprobs` (the teacher's top-k
next-token distribution at every position). We map those token strings to ids
using DeepSeek's own tokenizer, then save aligned arrays that distill.py trains
the student on with a KL soft-target loss.

    export DEEPSEEK_API_KEY=sk-...
    pip install transformers                     # for the DeepSeek tokenizer
    python scripts/distill_collect.py --num_docs=200 --out=out/distill.pt
    python distill.py --data=out/distill.pt

Each saved position t stores: the token id, and the teacher's top-k (ids,
logprobs) that produced it. distill.py aligns position t's distribution with the
student's prediction of token t. See `--dry_run` to preview without API calls.

NOTE: the API returns up to ~20 top_logprobs (not the full vocab), so this is
*approximate* (top-k) distillation. Token-string -> id mapping is best-effort;
positions that don't resolve cleanly are masked out.
"""

import argparse
import os
import random
import sys
import time

import requests
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import HFTokenizer  # noqa: E402

DEFAULT_BASE_URL = "https://api.deepseek.com"


def call_with_logprobs(base_url, api_key, model, messages, max_tokens, top_logprobs,
                       temperature, retries=4):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "logprobs": True, "top_logprobs": top_logprobs,
        "stream": False,
    }
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"DeepSeek API {resp.status_code}: {resp.text[:300]}")
            print(f"  retry {attempt + 1}/{retries} (HTTP {resp.status_code}) ...")
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}/{retries} ({e}) ...")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("DeepSeek API failed after retries.")


def parse_logprobs(choice, tokenizer, K):
    """Turn one API response into aligned (ids, topk_ids, topk_logprobs, mask)."""
    content = choice.get("logprobs", {}).get("content") or []
    ids, topk_ids, topk_lp, mask = [], [], [], []
    for entry in content:
        tid = tokenizer.token_to_id(entry["token"])
        if tid is None:
            continue  # unresolved token -> skip this position
        ids.append(tid)

        row_ids, row_lp, row_mask = [], [], []
        for alt in (entry.get("top_logprobs") or [])[:K]:
            aid = tokenizer.token_to_id(alt["token"])
            if aid is None:
                continue
            row_ids.append(aid)
            row_lp.append(alt["logprob"])
            row_mask.append(1.0)
        # Pad each row to width K.
        while len(row_ids) < K:
            row_ids.append(0)
            row_lp.append(-1e4)
            row_mask.append(0.0)
        topk_ids.append(row_ids)
        topk_lp.append(row_lp)
        mask.append(row_mask)
    return ids, topk_ids, topk_lp, mask


def build_messages(topic, prompt, nonce):
    system = ("You are a prolific writer generating clean, self-contained training "
              "text. Write natural prose only. Each response is one standalone passage.")
    if prompt:
        user = f"{prompt}\n\n(Variation #{nonce}.)"
    else:
        user = (f"Write a unique self-contained passage (a few paragraphs) about: "
                f"{topic}. Vary style and wording. Variation #{nonce}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a distillation dataset from DeepSeek.")
    parser.add_argument("--topic", default="short stories and conversations")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--top_logprobs", type=int, default=20, help="teacher top-k (max 20)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tokenizer_repo", default=HFTokenizer.DEFAULT_REPO)
    parser.add_argument("--out", default="out/distill.pt")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    K = min(args.top_logprobs, 20)

    if args.dry_run:
        import json
        print(json.dumps(build_messages(args.topic, args.prompt, 1), indent=2))
        print(f"\n[dry run] {args.num_docs} docs, top_logprobs={K}, model={args.model}")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("Set DEEPSEEK_API_KEY (https://platform.deepseek.com).")

    print(f"Loading DeepSeek tokenizer ({args.tokenizer_repo}) ...")
    tokenizer = HFTokenizer(args.tokenizer_repo)

    all_ids, all_topk_ids, all_topk_lp, all_mask = [], [], [], []
    for i in range(args.num_docs):
        nonce = random.randint(0, 1_000_000)
        choice = call_with_logprobs(
            args.base_url, api_key, args.model,
            build_messages(args.topic, args.prompt, nonce),
            args.max_tokens, K, args.temperature,
        )
        ids, topk_ids, topk_lp, mask = parse_logprobs(choice, tokenizer, K)
        all_ids += ids
        all_topk_ids += topk_ids
        all_topk_lp += topk_lp
        all_mask += mask
        print(f"[{i + 1}/{args.num_docs}] +{len(ids)} positions (total {len(all_ids):,})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "tokenizer": "deepseek",
        "tokenizer_repo": args.tokenizer_repo,
        "vocab_size": tokenizer.vocab_size,
        "K": K,
        "ids": torch.tensor(all_ids, dtype=torch.long),
        "topk_ids": torch.tensor(all_topk_ids, dtype=torch.long),
        "topk_logprobs": torch.tensor(all_topk_lp, dtype=torch.float32),
        "mask": torch.tensor(all_mask, dtype=torch.float32),
    }, args.out)
    print(f"\nSaved {len(all_ids):,} positions to {args.out} "
          f"(vocab {tokenizer.vocab_size}). Train with: python distill.py --data={args.out}")


if __name__ == "__main__":
    main()
