"""Batch-level convenience wrapper around MimiCodec.encode for CLI/dataset use."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch

from codec.current.wrapper import SAMPLE_RATE, MimiCodec


def encode_file(codec: MimiCodec, path: Path) -> torch.LongTensor:
    waveform, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr}Hz, expected {SAMPLE_RATE}Hz -- resample first")
    return codec.encode(waveform, sr)
