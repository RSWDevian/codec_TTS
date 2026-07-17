"""Character-level (grapheme) tokenizer for the toy prototype."""

from __future__ import annotations

import json
from pathlib import Path

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]


class GraphemeTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.pad_id = self.char_to_id[PAD]
        self.bos_id = self.char_to_id[BOS]
        self.eos_id = self.char_to_id[EOS]

    @classmethod
    def build_from_texts(cls, texts: list[str]) -> "GraphemeTokenizer":
        chars = sorted(set("".join(texts)))
        return cls(SPECIAL_TOKENS + chars)

    @classmethod
    def load(cls, path: str | Path) -> "GraphemeTokenizer":
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.bos_id] + [self.char_to_id[c] for c in text] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.vocab[i] for i in ids if self.vocab[i] not in SPECIAL_TOKENS
        )
