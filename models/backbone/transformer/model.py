"""Toy backbone: predicts the codebook-0 (semantic) token per audio frame,
conditioned on the text. Sequence = [text tokens][audio codebook-0 tokens],
fed as inputs_embeds (text and audio use separate embedding tables) into a
small causal LlamaModel, so text and audio can share one transformer despite
having different vocabularies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaModel

from models.backbone.transformer.config import ToyBackboneConfig
from models.embeddings.codec_embeddings import MimiCodebookEmbeddings
from models.heads.lm_head import LMHead


class ToyBackbone(nn.Module):
    def __init__(self, config: ToyBackboneConfig, codec_embeddings: MimiCodebookEmbeddings):
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.codec_embeddings = codec_embeddings
        self.llama = LlamaModel(config.to_llama_config())
        self.lm_head = LMHead(config.hidden_size, config.audio_vocab_size)

    def forward(
        self,
        text_ids: torch.LongTensor,
        audio_codebook0_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """text_ids: (B, Tt), audio_codebook0_ids: (B, Ta).

        Returns (audio_hidden [B, Ta, H], logits [B, Ta, audio_vocab_size]).
        logits[:, t] predicts audio_codebook0_ids[:, t+1] (teacher forcing);
        the caller is responsible for the shift when computing the loss.
        """
        text_emb = self.text_embedding(text_ids)
        audio_emb = self.codec_embeddings.embed_codebook(0, audio_codebook0_ids)
        inputs_embeds = torch.cat([text_emb, audio_emb], dim=1)
        outputs = self.llama(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        audio_hidden = hidden[:, text_ids.shape[1]:, :]
        logits = self.lm_head(audio_hidden)
        return audio_hidden, logits
