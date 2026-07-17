"""Composes resample -> normalize -> Mimi encode for one LJSpeech utterance."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from codec.current.wrapper import MimiCodec
from preprocessing.audio.normalize import peak_normalize
from preprocessing.audio.resample import resample


def tokenize_utterance(
    codec: MimiCodec, audio_path: str, native_sample_rate: int, target_sample_rate: int
) -> torch.LongTensor:
    waveform, sr = sf.read(audio_path, dtype="float32")
    if sr != native_sample_rate:
        raise ValueError(f"{audio_path} is {sr}Hz, expected {native_sample_rate}Hz")
    waveform = resample(waveform, sr, target_sample_rate)
    waveform = peak_normalize(waveform)
    codes = codec.encode(waveform, target_sample_rate)
    return codes[0]  # drop batch dim -> (NUM_CODEBOOKS, num_frames)
