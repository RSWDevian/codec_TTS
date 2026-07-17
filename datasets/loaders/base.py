"""Common interface implemented by every dataset loader under datasets/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSDatasetLoader(ABC):
    @abstractmethod
    def list_utterances(self) -> list[str]:
        """Return all utterance ids available in this dataset."""

    @abstractmethod
    def get_audio_path(self, utt_id: str) -> Path:
        """Return the path to the raw audio file for `utt_id`."""

    @abstractmethod
    def get_text(self, utt_id: str) -> str:
        """Return the transcript text for `utt_id`."""
