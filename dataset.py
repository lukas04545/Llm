"""
Data loading.

Three sources, increasing in scale:

  1. In-memory      -- tokenize a text file into one tensor. Best for tiny data.
  2. Memory-mapped  -- read pre-tokenized uint16 `.bin` files via mmap, so the
                       corpus never has to fit in RAM. Best for large local data.
  3. Streaming web  -- stream a Hugging Face web-scale dataset (FineWeb, C4,
                       OpenWebText, ...), tokenizing on the fly. Lets the model
                       train on an effectively unbounded stream of internet text
                       without downloading it. Requires the `datasets` package.
"""

import mmap

import torch

from tokenizer import build_tokenizer, load_tokenizer


# --------------------------------------------------------------------------- #
# 1. In-memory
# --------------------------------------------------------------------------- #
def load_corpus(path: str, val_split: float, tokenizer_kind: str = "char",
                vocab_size: int = 512):
    """Read text, build a tokenizer, and split into train/val id tensors."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = build_tokenizer(tokenizer_kind, text, vocab_size)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n_val = int(len(data) * val_split)
    return tokenizer, data[: len(data) - n_val], data[len(data) - n_val :]


def get_batch(data, block_size, batch_size, device):
    """Sample a random batch of (input, target) sequences from a token tensor.

    Targets are inputs shifted by one position. Works for both an in-memory
    LongTensor and a memmap-backed uint16 tensor (the batch is cast to long).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).long()

    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --------------------------------------------------------------------------- #
# 2. Memory-mapped binary
# --------------------------------------------------------------------------- #
def write_bin(ids, path: str) -> None:
    """Write a list/iterable of token ids to a uint16 binary file."""
    import array
    arr = array.array("H", ids)  # 'H' = unsigned short (uint16)
    with open(path, "wb") as f:
        arr.tofile(f)


def load_bin(path: str) -> torch.Tensor:
    """Memory-map a uint16 `.bin` file as a 1-D tensor (no full RAM load)."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # frombuffer shares memory with the mmap; the OS pages it in on demand.
    return torch.frombuffer(mm, dtype=torch.uint16)


# --------------------------------------------------------------------------- #
# 3. Streaming web-scale source
# --------------------------------------------------------------------------- #
# Friendly preset names -> (hf path, config, split, text field).
STREAM_PRESETS = {
    "fineweb":     ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text"),
    "c4":          ("allenai/c4", "en", "train", "text"),
    "openwebtext": ("Skylion007/openwebtext", None, "train", "text"),
    "wikitext":    ("wikitext", "wikitext-103-raw-v1", "train", "text"),
}


class StreamingDataset:
    """Stream + tokenize a Hugging Face dataset into training batches.

    Documents are tokenized on the fly and concatenated (separated by an
    end-of-text id) into a rolling buffer; fixed `block_size`+1 windows are
    sliced out to form batches. Nothing is held beyond the current buffer.
    """

    def __init__(self, preset_or_path, tokenizer, block_size, batch_size, device,
                 hf_config=None, split="train", text_field="text", seed=1337):
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Streaming web data needs the 'datasets' package. "
                "Install with: pip install datasets"
            ) from e

        if preset_or_path in STREAM_PRESETS:
            path, hf_config, split, text_field = STREAM_PRESETS[preset_or_path]
        else:
            path = preset_or_path

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.text_field = text_field
        # A newline is a safe document separator for every tokenizer kind here.
        self.sep_ids = tokenizer.encode("\n")

        ds = load_dataset(path, hf_config, split=split, streaming=True)
        self.ds = ds.shuffle(seed=seed, buffer_size=10_000)
        self._iter = iter(self.ds)
        self._buf: list[int] = []

    def _refill(self, need: int) -> None:
        while len(self._buf) < need:
            try:
                doc = next(self._iter)
            except StopIteration:
                self._iter = iter(self.ds)  # loop the (effectively endless) stream
                doc = next(self._iter)
            text = doc.get(self.text_field, "")
            if text:
                self._buf.extend(self.tokenizer.encode(text))
                self._buf.extend(self.sep_ids)

    def get_batch(self):
        span = self.block_size + 1
        need = self.batch_size * span
        self._refill(need)

        xs, ys = [], []
        for b in range(self.batch_size):
            chunk = self._buf[b * span : b * span + span]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        # Drop the consumed tokens (keep the remainder for the next batch).
        self._buf = self._buf[self.batch_size * span :]

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
