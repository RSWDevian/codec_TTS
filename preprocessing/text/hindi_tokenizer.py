"""Devanagari character-level tokenizer for Hindi TTS.

Builds a vocabulary from Devanagari Unicode range + common punctuation.
Uses indic_transliteration for normalization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
DIGIT_RE = re.compile(r"[\u0966-\u096F]")
PUNCTUATION = set(" ।,?!;:\"'()-|/.")


class HindiTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.pad_id = self.char_to_id[PAD]
        self.bos_id = self.char_to_id[BOS]
        self.eos_id = self.char_to_id[EOS]

    @classmethod
    def build_from_texts(cls, texts: list[str]) -> "HindiTokenizer":
        chars: set[str] = set()
        for text in texts:
            for c in text:
                if DEVANAGARI_RE.match(c) or DIGIT_RE.match(c) or c in PUNCTUATION or c in " \n":
                    chars.add(c)
        sorted_chars = sorted(chars, key=lambda c: ord(c))
        return cls(SPECIAL_TOKENS + sorted_chars)

    @classmethod
    def load(cls, path: str | Path) -> "HindiTokenizer":
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_id]
        for c in text:
            if c in self.char_to_id:
                ids.append(self.char_to_id[c])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.vocab[i] for i in ids if self.vocab[i] not in SPECIAL_TOKENS
        )
