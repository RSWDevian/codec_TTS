"""Resample audio to the codec's expected sample rate (24kHz for Mimi)."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()
