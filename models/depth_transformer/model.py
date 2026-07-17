"""Toy depth transformer: given the backbone's hidden state for one audio
frame, autoregressively predicts codebooks 1..7 for that frame (teacher
forced). Frames are independent of each other here -- each is a length-7
sequence [projected backbone hidden state, codebook_1, ..., codebook_6],
flattened over batch*time into the "batch" dimension of a small causal
LlamaModel, matching the Moshi/CSM per-frame depth-transformer design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaModel

from models.depth_transformer.config import ToyDepthConfig
from models.embeddings.codec_embeddings import MimiCodebookEmbeddings
from models.heads.lm_head import LMHead


class ToyDepthTransformer(nn.Module):
    def __init__(self, config: ToyDepthConfig, codec_embeddings: MimiCodebookEmbeddings):
        super().__init__()
        self.config = config
        self.backbone_proj = nn.Linear(config.backbone_hidden_size, config.hidden_size)
        self.codec_embeddings = codec_embeddings
        codec_dim = codec_embeddings.embeddings[0].embedding_dim
        self.codec_in_proj = nn.Linear(codec_dim, config.hidden_size)
        self.llama = LlamaModel(config.to_llama_config())
        self.lm_head = LMHead(config.hidden_size, config.audio_vocab_size)

    def forward(
        self, backbone_hidden: torch.Tensor, codebooks_1_to_6_ids: torch.LongTensor
    ) -> torch.Tensor:
        """backbone_hidden: (N, backbone_hidden_size) -- N = flattened batch*time.
        codebooks_1_to_6_ids: (N, 6) teacher-forced tokens for codebooks 1..6.

        Returns logits (N, 7, audio_vocab_size) predicting codebooks 1..7.
        """
        proj_hidden = self.backbone_proj(backbone_hidden).unsqueeze(1)  # (N, 1, H)
        codebook_embeds = [
            self.codec_in_proj(self.codec_embeddings.embed_codebook(i + 1, codebooks_1_to_6_ids[:, i]))
            for i in range(codebooks_1_to_6_ids.shape[1])
        ]
        inputs_embeds = torch.cat([proj_hidden, torch.stack(codebook_embeds, dim=1)], dim=1)  # (N, 7, H)
        outputs = self.llama(inputs_embeds=inputs_embeds)
        return self.lm_head(outputs.last_hidden_state)  # (N, 7, vocab)
