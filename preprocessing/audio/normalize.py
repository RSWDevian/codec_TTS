"""Simple peak normalization for the toy prototype."""

from __future__ import annotations

import numpy as np


def peak_normalize(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.abs(waveform).max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_peak / peak)
