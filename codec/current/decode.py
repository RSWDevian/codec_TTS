"""Batch-level convenience wrapper around MimiCodec.decode, writes .wav files."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch

from codec.current.wrapper import SAMPLE_RATE, MimiCodec


def decode_to_file(codec: MimiCodec, codes: torch.LongTensor, path: Path) -> None:
    audio = codec.decode(codes)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE)
