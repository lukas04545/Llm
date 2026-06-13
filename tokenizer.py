"""
Tokenizers.

Three options, all behind a shared interface (`encode`, `decode`, `vocab_size`,
`save`, `load`):

  * CharTokenizer  -- character level, zero dependencies. Great for tiny corpora.
  * BPETokenizer   -- byte-level Byte-Pair Encoding trained from scratch (no
                      external library). Fewer tokens per text -> less compute.
  * GPT2Tokenizer  -- GPT-2's pretrained 50k BPE via `tiktoken` (optional dep).
                      Handles arbitrary internet text out of the box; ideal when
                      streaming web-scale corpora.

Use `load_tokenizer(path)` to restore whichever kind was saved.
"""

import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# Character level
# --------------------------------------------------------------------------- #
class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str | Path) -> None:
        _write_json(path, {"kind": self.kind, "chars": self.chars})

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(_read_json(path)["chars"])


# --------------------------------------------------------------------------- #
# Byte-level BPE (trained from scratch)
# --------------------------------------------------------------------------- #
class BPETokenizer:
    """A minimal byte-level Byte-Pair Encoding tokenizer.

    Bytes (0-255) are the base vocabulary; merges are learned greedily by
    repeatedly fusing the most frequent adjacent pair. This is the same core
    algorithm GPT-2 uses, implemented compactly with no dependencies.
    """

    kind = "bpe"

    def __init__(self, merges: dict[tuple[int, int], int], vocab_size: int):
        self.merges = merges                      # (a, b) -> new_id
        self._vocab_size = vocab_size
        # Build the id -> bytes table for decoding.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256, "vocab_size must be >= 256 for byte-level BPE"
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = _pair_counts(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            merges[pair] = new_id
            if verbose and (i % 100 == 0 or i == num_merges - 1):
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id}")
        return cls(merges, 256 + len(merges))

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _pair_counts(ids)
            # Merge the pair with the lowest merge index that still exists.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab.get(i, b"") for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        # JSON keys must be strings; encode pairs as "a,b".
        merges = {f"{a},{b}": idx for (a, b), idx in self.merges.items()}
        _write_json(path, {"kind": self.kind, "merges": merges,
                           "vocab_size": self._vocab_size})

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = _read_json(path)
        merges = {}
        for key, idx in data["merges"].items():
            a, b = key.split(",")
            merges[(int(a), int(b))] = idx
        return cls(merges, data["vocab_size"])


def _pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# --------------------------------------------------------------------------- #
# GPT-2 BPE via tiktoken (optional; ideal for web-scale streaming)
# --------------------------------------------------------------------------- #
class GPT2Tokenizer:
    """Wraps GPT-2's pretrained 50257-token BPE. Requires `tiktoken`."""

    kind = "gpt2"

    def __init__(self):
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'gpt2' tokenizer needs tiktoken. Install with: pip install tiktoken"
            ) from e
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def save(self, path: str | Path) -> None:
        _write_json(path, {"kind": self.kind})

    @classmethod
    def load(cls, path: str | Path) -> "GPT2Tokenizer":
        return cls()


# --------------------------------------------------------------------------- #
# Hugging Face tokenizer (e.g. DeepSeek) -- for cross-model distillation
# --------------------------------------------------------------------------- #
class HFTokenizer:
    """Wraps a pretrained Hugging Face tokenizer. Requires `transformers`.

    Using the *teacher's* tokenizer (e.g. DeepSeek's) makes the student's token
    ids line up with the teacher's, which is what enables logprob distillation
    (see distill.py). Note the large vocab inflates the embedding table.
    """

    kind = "hf"
    DEFAULT_REPO = "deepseek-ai/DeepSeek-V3"

    def __init__(self, repo: str | None = None):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'deepseek' tokenizer needs transformers. "
                "Install with: pip install transformers"
            ) from e
        self.repo = repo or self.DEFAULT_REPO
        self.tok = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=True)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def token_to_id(self, token: str):
        """Best-effort map an API logprob token string to a single id (or None)."""
        ids = self.tok.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        tid = self.tok.convert_tokens_to_ids(token)
        unk = self.tok.unk_token_id
        return tid if (tid is not None and tid != unk) else None

    def save(self, path: str | Path) -> None:
        self.save_config(path, self.repo)

    @staticmethod
    def save_config(path: str | Path, repo: str | None) -> None:
        """Write the tokenizer metadata without instantiating transformers."""
        _write_json(path, {"kind": HFTokenizer.kind, "repo": repo or HFTokenizer.DEFAULT_REPO})

    @classmethod
    def load(cls, path: str | Path) -> "HFTokenizer":
        return cls(_read_json(path).get("repo"))


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str | Path):
    """Restore a tokenizer of whatever kind was saved at `path`."""
    kind = _read_json(path).get("kind", "char")
    registry = {"char": CharTokenizer, "bpe": BPETokenizer, "gpt2": GPT2Tokenizer,
                "hf": HFTokenizer, "deepseek": HFTokenizer}  # "deepseek" kept for back-compat
    return registry[kind].load(path)


def build_tokenizer(kind: str, text: str | None, vocab_size: int):
    """Construct (and, for char/bpe, fit) a tokenizer of the requested kind."""
    if kind == "char":
        return CharTokenizer.from_text(text or "")
    if kind == "bpe":
        return BPETokenizer.train(text or "", vocab_size)
    if kind == "gpt2":
        return GPT2Tokenizer()
    if kind in ("hf", "deepseek"):
        return HFTokenizer()
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def _write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _read_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
