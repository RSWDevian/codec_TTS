"""Per-codebook embedding tables shared between the backbone and depth transformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from codec.current.wrapper import CODEBOOK_SIZE, NUM_CODEBOOKS


class MimiCodebookEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_codebooks: int = NUM_CODEBOOKS,
        codebook_size: int = CODEBOOK_SIZE,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_size) for _ in range(num_codebooks)]
        )

    def embed_codebook(self, codebook_idx: int, ids: torch.LongTensor) -> torch.Tensor:
        return self.embeddings[codebook_idx](ids)
