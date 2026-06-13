"""
Train the small GPT by distilling from a DeepSeek top-k logprob dataset.

This is the training half of cross-model distillation. Collect a dataset first:

    python scripts/distill_collect.py --num_docs=200 --out=out/distill.pt

then distill:

    python distill.py --data=out/distill.pt --max_iters=3000
    python generate.py --prompt="..."        # uses the saved DeepSeek tokenizer

The loss blends the usual hard-label cross-entropy with a KL term that pulls the
student's distribution toward the teacher's top-k soft targets:

    loss = alpha * CE(student, next_token)
         + (1 - alpha) * T^2 * KL(teacher_topk || student_topk)

Soft targets carry more signal than hard labels alone (they say *how* confident
the teacher was and what the runner-up tokens were). Because the API only gives
the top-k (not the full vocab), the KL is computed over the teacher's top-k ids.
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from config import TrainConfig, parse_overrides
from model import GPT, GPTConfig
from tokenizer import HFTokenizer
from train import get_lr, resolve_device, resolve_dtype


def get_distill_batch(data, block_size, batch_size, device):
    """Sample windows of (x, hard targets, teacher top-k ids/logprobs/mask).

    Teacher distributions are aligned to the token they produced, so the targets
    for predicting tokens x[1:] are the top-k arrays at the same shifted offset.
    """
    ids = data["ids"]
    n = len(ids) - block_size - 1
    ix = torch.randint(max(1, n), (batch_size,))
    x = torch.stack([ids[i : i + block_size] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + block_size] for i in ix])
    t_ids = torch.stack([data["topk_ids"][i + 1 : i + 1 + block_size] for i in ix])
    t_lp = torch.stack([data["topk_logprobs"][i + 1 : i + 1 + block_size] for i in ix])
    t_mask = torch.stack([data["mask"][i + 1 : i + 1 + block_size] for i in ix])
    tensors = [t.to(device) for t in (x, y, t_ids, t_lp, t_mask)]
    return tensors


def distill_loss(logits, y, topk_ids, topk_lp, topk_mask, alpha, temp):
    """Combined hard-label CE + top-k KL soft-target loss."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))

    # Student log-probs restricted to the teacher's top-k ids at each position.
    student_topk = torch.gather(logits, -1, topk_ids) / temp          # (B,T,K)
    student_logp = F.log_softmax(student_topk, dim=-1)

    # Teacher distribution over the same top-k (renormalized; padding masked out).
    teacher = F.softmax(topk_lp / temp, dim=-1) * topk_mask
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-9)

    kl = (teacher * (teacher.clamp_min(1e-9).log() - student_logp)).sum(-1).mean()
    loss = alpha * ce + (1 - alpha) * (temp ** 2) * kl
    return loss, ce, kl


def main(argv: list[str]) -> None:
    # Reuse the standard config; add a couple of distillation-only knobs.
    cfg = TrainConfig()
    cfg.data_path = ""  # unused here
    alpha, temp, data_path = 0.5, 1.0, "out/distill.pt"
    extra = []
    for arg in argv:
        if arg.startswith("--alpha="):
            alpha = float(arg.split("=", 1)[1])
        elif arg.startswith("--temp="):
            temp = float(arg.split("=", 1)[1])
        elif arg.startswith("--data="):
            data_path = arg.split("=", 1)[1]
        else:
            extra.append(arg)
    cfg = parse_overrides(cfg, extra)

    device = resolve_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = resolve_dtype(cfg.dtype, device_type)
    print(f"Device: {device} | dtype: {str(dtype).split('.')[-1]} | "
          f"alpha(CE)={alpha} temp={temp}")

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    if not os.path.exists(data_path):
        sys.exit(f"No distill dataset at {data_path}. Run scripts/distill_collect.py first.")
    data = torch.load(data_path, map_location="cpu")
    vocab_size = data["vocab_size"]
    print(f"Distill data: {len(data['ids']):,} positions | vocab {vocab_size} "
          f"| teacher top-{data['K']}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    # Persist the matching tokenizer metadata so generate.py can decode later.
    # (Writing config only -- no need to load transformers during training.)
    HFTokenizer.save_config(os.path.join(cfg.out_dir, "tokenizer.json"),
                            data.get("tokenizer_repo"))

    model_args = dict(
        vocab_size=vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd,
        dropout=cfg.dropout, bias=cfg.bias, norm=cfg.norm, mlp=cfg.mlp,
        use_rope=cfg.use_rope,
    )
    model = GPT(GPTConfig(**model_args)).to(device)
    model.grad_checkpoint = cfg.grad_checkpoint
    print(f"Student parameters: {model.num_params() / 1e6:.2f}M")

    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type,
        optimizer=cfg.optimizer,
    )
    use_amp = device_type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    amp_ctx = (torch.autocast(device_type="cuda", dtype=dtype) if use_amp
               else _null())
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(it, cfg)

        if it % cfg.log_interval == 0 or it == cfg.max_iters:
            model.eval()
            with torch.no_grad():
                x, y, ti, tl, tm = get_distill_batch(data, cfg.block_size, cfg.batch_size, device)
                with amp_ctx:
                    logits, _ = model(x, y)
                    loss, ce, kl = distill_loss(logits, y, ti, tl, tm, alpha, temp)
            dt = (time.time() - t0) * 1000 / max(1, cfg.log_interval)
            print(f"step {it:5d} | loss {loss.item():.4f} | CE {ce.item():.4f} "
                  f"| KL {kl.item():.4f} | {dt:.1f} ms/iter")
            t0 = time.time()
            torch.save({"model": model.state_dict(), "model_args": model_args,
                        "iter": it}, os.path.join(cfg.out_dir, "ckpt.pt"))
            model.train()

        if it == cfg.max_iters:
            break

        for _ in range(cfg.grad_accum_steps):
            x, y, ti, tl, tm = get_distill_batch(data, cfg.block_size, cfg.batch_size, device)
            with amp_ctx:
                logits, _ = model(x, y)
                loss, _, _ = distill_loss(logits, y, ti, tl, tm, alpha, temp)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    print(f"\nDone. Checkpoint: {os.path.join(cfg.out_dir, 'ckpt.pt')}")
    print("Generate with:  python generate.py --prompt='...'")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main(sys.argv[1:])
