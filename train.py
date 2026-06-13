"""
Train the small GPT.

Usage (local or Colab):
    python train.py
    python train.py --data_path=data/input.txt --max_iters=5000 --batch_size=64

Checkpoints and the tokenizer are written to `out_dir` (default: ./out).
Generation can then be run with `python generate.py`.
"""

import math
import os
import sys
import time

import torch

from config import TrainConfig, parse_overrides
from dataset import get_batch, load_corpus
from model import GPT, GPTConfig


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine learning-rate schedule with linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, datasets, cfg, device):
    """Average loss over a few batches for train and val splits."""
    model.eval()
    out = {}
    for split, data in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(argv: list[str]) -> None:
    cfg = parse_overrides(TrainConfig(), argv)
    device = resolve_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else device
    print(f"Using device: {device}")

    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- data ---
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"Training data not found at '{cfg.data_path}'. "
            f"Provide a UTF-8 text file via --data_path=..."
        )
    tokenizer, train_data, val_data = load_corpus(cfg.data_path, cfg.val_split)
    datasets = {"train": train_data, "val": val_data}
    print(f"Corpus: {len(train_data) + len(val_data):,} tokens | "
          f"vocab size: {tokenizer.vocab_size}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(cfg.out_dir, "tokenizer.json"))

    # --- model ---
    model_args = dict(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(GPTConfig(**model_args)).to(device)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M")

    if cfg.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )

    # Mixed precision on CUDA for speed + lower memory.
    use_amp = device_type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _nullcontext()
    )

    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic evaluation + checkpointing.
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, datasets, cfg, device)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": getattr(model, "_orig_mod", model).state_dict(),
                    "model_args": model_args,
                    "iter": it,
                    "best_val_loss": best_val_loss,
                    "config": vars(cfg),
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "ckpt.pt"))

        if it == cfg.max_iters:
            break

        # --- one optimization step ---
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        with amp_ctx:
            _, loss = model(x, y)
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
            print(f"step {it:5d} | loss {loss.item():.4f} | "
                  f"{dt * 1000 / max(1, cfg.log_interval):.1f} ms/iter")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {os.path.join(cfg.out_dir, 'ckpt.pt')}")
    print("Generate text with:  python generate.py --prompt='...'")


class _nullcontext:
    """Tiny no-op context manager (kept local to avoid extra imports)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main(sys.argv[1:])
