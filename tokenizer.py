"""
A simple character-level tokenizer.

Character-level tokenization keeps the project dependency-free (no external
tokenizer library) and makes the model easy to train on small datasets. The
vocabulary is built directly from the characters present in the training text
and saved to a JSON file so encoding/decoding stays consistent across runs.
"""

import json
from pathlib import Path


class CharTokenizer:
    """Maps characters to integer ids and back."""

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from the unique characters in `text`."""
        chars = sorted(set(text))
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        # Unknown characters are skipped (they cannot appear if the tokenizer
        # was built from the same corpus).
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])
