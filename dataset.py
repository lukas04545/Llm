"""
Data loading utilities.

The whole corpus is tokenized once and held in memory as a single long tensor
of ids. Training batches are sampled by picking random offsets into that
tensor — simple and fast, and a good fit for small datasets.
"""

import torch

from tokenizer import CharTokenizer


def load_corpus(path: str, val_split: float):
    """Read text, build a tokenizer, and split into train/val id tensors."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = len(data)
    n_val = int(n * val_split)
    train_data = data[: n - n_val]
    val_data = data[n - n_val :]
    return tokenizer, train_data, val_data


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences.

    Targets are inputs shifted by one position — the model predicts the next
    token at every position.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    if device.startswith("cuda"):
        # Pin + async transfer for a small speedup on GPU.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
