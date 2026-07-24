"""Per-codebook embedding tables shared between the backbone and depth transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
from codec.current.wrapper import CODEBOOK_SIZE, NUM_CODEBOOKS

# Codebook 0 (the backbone's stream) reserves one extra id past the real Mimi
# codebook range to mark end-of-audio, so generation length is model-driven
# instead of a fixed frame-count heuristic. Codebooks 1..7 (depth transformer)
# never need this -- by the time the depth transformer runs on a frame, the
# backbone has already decided that frame is real.
AUDIO_EOS_ID = CODEBOOK_SIZE


class MimiCodebookEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_codebooks: int = NUM_CODEBOOKS,
        codebook_size: int = CODEBOOK_SIZE,
    ):
        super().__init__()
        sizes = [codebook_size + 1] + [codebook_size] * (num_codebooks - 1)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_size) for size in sizes]
        )

    def embed_codebook(self, codebook_idx: int, ids: torch.LongTensor) -> torch.Tensor:
        return self.embeddings[codebook_idx](ids)
