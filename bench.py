"""
Benchmark generation speed, memory, and model size.

Reports tokens/second, peak RAM (CPU) or peak GPU memory, and on-disk/in-memory
model size for a trained checkpoint under a chosen quantization mode and with or
without the KV cache. Use it to compare configurations, e.g.:

    python bench.py --quantize=none
    python bench.py --quantize=int8
    python bench.py --quantize=int4 --device=cuda
    python bench.py --no_cache            # see the speedup the KV cache gives
"""

import argparse
import time
import tracemalloc

import torch

from generate import load_model, resolve_device
from quantize import model_size_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained small GPT.")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantize", default="none",
                        choices=["none", "int8", "int4", "fp4", "nf4"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.quantize == "int8":
        device = "cpu"
    use_cuda = device.startswith("cuda")

    model, tokenizer = load_model(args.out_dir, device, args.quantize)
    start_ids = tokenizer.encode(args.prompt) or tokenizer.encode("\n")
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    def run_once():
        return model.generate(idx, args.max_new_tokens, temperature=0.8,
                              top_k=40, use_cache=not args.no_cache)

    for _ in range(args.warmup):
        run_once()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()

    times = []
    for _ in range(args.runs):
        t0 = time.time()
        run_once()
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    tok_per_s = args.max_new_tokens / avg

    print("\n=== benchmark ===")
    print(f"device:           {device}")
    print(f"quantize:         {args.quantize}")
    print(f"KV cache:         {'off' if args.no_cache else 'on'}")
    print(f"model size:       {model_size_mb(model):.2f} MB")
    print(f"new tokens:       {args.max_new_tokens}")
    print(f"avg time:         {avg * 1000:.1f} ms  (over {args.runs} runs)")
    print(f"throughput:       {tok_per_s:.1f} tokens/sec")
    if use_cuda:
        print(f"peak GPU memory:  {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    else:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"peak CPU alloc:   {peak / 1024**2:.1f} MB (generation only)")


if __name__ == "__main__":
    main()
