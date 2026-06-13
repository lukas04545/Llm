"""
Train the small GPT.

Usage (local or Colab):
    python train.py
    python train.py --max_iters=5000 --batch_size=64 --grad_accum_steps=4
    python train.py --dtype=bf16 --grad_checkpoint=true --compile=true
    python train.py --tokenizer=bpe --vocab_size=1024

Train on web-scale streaming data (no download, effectively endless):
    python train.py --stream=fineweb --tokenizer=gpt2 --device=cuda \
        --n_layer=6 --n_head=6 --n_embd=384 --block_size=256

Checkpoints + tokenizer are written to `out_dir` (default: ./out).
"""

import math
import os
import sys
import time
from contextlib import nullcontext

import torch

from config import TrainConfig, parse_overrides
from dataset import StreamingDataset, get_batch, load_bin, load_corpus
from model import GPT, GPTConfig
from tokenizer import build_tokenizer, load_tokenizer


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device_type: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    # auto
    if device_type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine learning-rate schedule with linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def build_data(cfg: TrainConfig, device: str):
    """Return (tokenizer, train_batch_fn, val_batch_fn) for the chosen source."""
    # --- streaming web-scale source ---
    if cfg.stream:
        tokenizer = _streaming_tokenizer(cfg)
        hf_config = cfg.hf_config or None
        train_src = StreamingDataset(cfg.stream, tokenizer, cfg.block_size,
                                     cfg.batch_size, device, hf_config=hf_config,
                                     split=cfg.split, text_field=cfg.text_field,
                                     seed=cfg.seed)
        val_src = StreamingDataset(cfg.stream, tokenizer, cfg.block_size,
                                   cfg.batch_size, device, hf_config=hf_config,
                                   split=cfg.split, text_field=cfg.text_field,
                                   seed=cfg.seed + 1)
        print(f"Streaming source: {cfg.stream} | tokenizer: {tokenizer.kind} "
              f"| vocab size: {tokenizer.vocab_size}")
        return tokenizer, train_src.get_batch, val_src.get_batch

    # --- pre-tokenized memmap .bin source ---
    train_bin = os.path.join(cfg.out_dir, "train.bin")
    val_bin = os.path.join(cfg.out_dir, "val.bin")
    if os.path.exists(train_bin) and os.path.exists(val_bin):
        tokenizer = load_tokenizer(os.path.join(cfg.out_dir, "tokenizer.json"))
        train_data, val_data = load_bin(train_bin), load_bin(val_bin)
        print(f"Memmap source: {train_bin} ({len(train_data):,} tokens) "
              f"| vocab size: {tokenizer.vocab_size}")
        return (tokenizer,
                lambda: get_batch(train_data, cfg.block_size, cfg.batch_size, device),
                lambda: get_batch(val_data, cfg.block_size, cfg.batch_size, device))

    # --- in-memory text source ---
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"Training data not found at '{cfg.data_path}'. Provide a UTF-8 text "
            f"file via --data_path=..., pre-tokenize with scripts/prepare_data.py, "
            f"or stream with --stream=fineweb."
        )
    tokenizer, train_data, val_data = load_corpus(
        cfg.data_path, cfg.val_split, cfg.tokenizer, cfg.vocab_size
    )
    print(f"Corpus: {len(train_data) + len(val_data):,} tokens | "
          f"tokenizer: {tokenizer.kind} | vocab size: {tokenizer.vocab_size}")
    return (tokenizer,
            lambda: get_batch(train_data, cfg.block_size, cfg.batch_size, device),
            lambda: get_batch(val_data, cfg.block_size, cfg.batch_size, device))


def _streaming_tokenizer(cfg: TrainConfig):
    """Build a tokenizer for streaming. gpt2 needs no fitting; char/bpe are
    fitted on a small sample drawn from the stream."""
    if cfg.tokenizer == "gpt2":
        return build_tokenizer("gpt2", None, cfg.vocab_size)

    from datasets import load_dataset
    print(f"Sampling from '{cfg.stream}' to fit the {cfg.tokenizer} tokenizer ...")
    path = cfg.stream
    hf_config = cfg.hf_config or None
    from dataset import STREAM_PRESETS
    if cfg.stream in STREAM_PRESETS:
        path, hf_config, _, _ = STREAM_PRESETS[cfg.stream]
    ds = load_dataset(path, hf_config, split=cfg.split, streaming=True)
    sample, total = [], 0
    for doc in ds:
        t = doc.get(cfg.text_field, "")
        if t:
            sample.append(t)
            total += len(t)
        if total >= 1_000_000:  # ~1 MB sample is plenty to learn merges
            break
    return build_tokenizer(cfg.tokenizer, "\n".join(sample), cfg.vocab_size)


class CUDAPrefetcher:
    """Prefetch the next batch on a side CUDA stream to overlap data movement
    (host->device copy) with model compute on the default stream."""

    def __init__(self, batch_fn):
        self.batch_fn = batch_fn
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self):
        # batch_fn already issues a pinned, non_blocking copy to the GPU; running
        # it inside the side stream lets that copy run concurrently with compute.
        with torch.cuda.stream(self.stream):
            self._next = self.batch_fn()

    def get_batch(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self._next
        # Keep the tensors alive until the default stream has consumed them.
        x.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        self._preload()
        return x, y


@torch.no_grad()
def estimate_loss(model, train_batch, val_batch, cfg, amp_ctx):
    model.eval()
    out = {}
    for split, batch_fn in (("train", train_batch), ("val", val_batch)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = batch_fn()
            with amp_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(argv: list[str]) -> None:
    cfg = parse_overrides(TrainConfig(), argv)
    device = resolve_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = resolve_dtype(cfg.dtype, device_type)
    print(f"Using device: {device} | dtype: {str(dtype).split('.')[-1]}")

    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # autotune kernels for fixed shapes
    torch.set_float32_matmul_precision("high")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # --- data ---
    tokenizer, train_batch, val_batch = build_data(cfg, device)
    tokenizer.save(os.path.join(cfg.out_dir, "tokenizer.json"))

    # --- model ---
    model_args = dict(
        vocab_size=tokenizer.vocab_size, block_size=cfg.block_size,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd, dropout=cfg.dropout, bias=cfg.bias,
        norm=cfg.norm, mlp=cfg.mlp, use_rope=cfg.use_rope,
    )
    model = GPT(GPTConfig(**model_args)).to(device)
    model.grad_checkpoint = cfg.grad_checkpoint
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M"
          + (" | grad checkpointing ON" if cfg.grad_checkpoint else ""))

    raw_model = model
    if cfg.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    optimizer = raw_model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type,
        optimizer=cfg.optimizer,
    )
    if cfg.optimizer == "adamw8bit":
        print("Using 8-bit AdamW optimizer (bitsandbytes)")

    # Async prefetch overlaps the next batch's H2D copy with current compute.
    if cfg.prefetch and device_type == "cuda":
        train_batch = CUDAPrefetcher(train_batch).get_batch

    # Mixed precision: autocast for bf16/fp16; GradScaler only for fp16.
    use_amp = device_type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    amp_ctx = (torch.autocast(device_type="cuda", dtype=dtype) if use_amp
               else nullcontext())
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_batch, val_batch, cfg, amp_ctx)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model": raw_model.state_dict(),
                    "model_args": model_args,
                    "iter": it,
                    "best_val_loss": best_val_loss,
                    "config": vars(cfg),
                }, os.path.join(cfg.out_dir, "ckpt.pt"))

        if it == cfg.max_iters:
            break

        # --- one optimization step (with gradient accumulation) ---
        for micro in range(cfg.grad_accum_steps):
            x, y = train_batch()
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens = cfg.batch_size * cfg.grad_accum_steps * cfg.block_size
            tok_per_s = tokens * cfg.log_interval / dt if it > 0 else 0
            print(f"step {it:5d} | loss {loss.item() * cfg.grad_accum_steps:.4f} | "
                  f"{dt * 1000 / max(1, cfg.log_interval):.1f} ms/iter"
                  + (f" | {tok_per_s/1e3:.1f}k tok/s" if tok_per_s else ""))

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {os.path.join(cfg.out_dir, 'ckpt.pt')}")
    print("Generate with:  python generate.py --prompt='...'")


if __name__ == "__main__":
    main(sys.argv[1:])
