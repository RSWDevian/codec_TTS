"""Loader for LJSpeech-1.1 (https://keithito.com/LJ-Speech-Dataset/).

Expects `root_dir` to contain `metadata.csv` (pipe-delimited: id|transcript|
normalized_transcript) and a `wavs/` subdirectory of `<id>.wav` files.
"""

from __future__ import annotations

import csv
from pathlib import Path

from datasets.loaders.base import TTSDatasetLoader


class LJSpeechLoader(TTSDatasetLoader):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self._transcripts: dict[str, str] = {}
        metadata_path = self.root_dir / "metadata.csv"
        with open(metadata_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in reader:
                utt_id, transcript, normalized = row[0], row[1], row[2] if len(row) > 2 else row[1]
                self._transcripts[utt_id] = normalized

    def list_utterances(self) -> list[str]:
        return list(self._transcripts.keys())

    def get_audio_path(self, utt_id: str) -> Path:
        return self.root_dir / "wavs" / f"{utt_id}.wav"

    def get_text(self, utt_id: str) -> str:
        return self._transcripts[utt_id]
