"""Shared contract for inference backends so scripts/inference.py can swap them."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePipeline(ABC):
    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> tuple[np.ndarray, int]:
        """Returns (audio_waveform, sample_rate)."""
