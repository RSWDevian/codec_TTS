"""Hindi dataset loader: reads a directory of paired .wav + .txt files.

Expected structure:
  data/raw/hindi/
    00001.wav
    00001.txt
    00002.wav
    00002.txt
    ...

Each .txt file contains Devanagari Hindi text.
"""

from __future__ import annotations
from pathlib import Path

class HindiDatasetLoader:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Hindi data directory not found: {root_dir}")

    def list_utterances(self) -> list[str]:
        wavs = sorted(self.root.glob("*.wav"))
        ids = []
        for w in wavs:
            txt_path = w.with_suffix(".txt")
            if txt_path.exists():
                ids.append(w.stem)
        return ids

    def get_audio_path(self, utt_id: str) -> Path:
        return self.root / f"{utt_id}.wav"

    def get_text(self, utt_id: str) -> str:
        path = self.root / f"{utt_id}.txt"
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
