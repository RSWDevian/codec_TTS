from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaModel

from models.backbone.transformer.config import HindiBackboneConfig
from models.embeddings.codec_embeddings import MimiCodebookEmbeddings
from models.heads.lm_head import LMHead


class HindiBackbone(nn.Module):
    def __init__(self, config: HindiBackboneConfig, codec_embeddings: MimiCodebookEmbeddings):
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.codec_embeddings = codec_embeddings
        self.audio_bos = nn.Parameter(torch.randn(config.hidden_size))
        self.llama = LlamaModel(config.to_llama_config())
        self.lm_head = LMHead(config.hidden_size, config.audio_vocab_size)

    def forward(
        self,
        text_ids: torch.LongTensor,
        audio_codebook0_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        add_bos: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_emb = self.text_embedding(text_ids)
        audio_emb = self.codec_embeddings.embed_codebook(0, audio_codebook0_ids)
        if add_bos:
            bos = self.audio_bos.view(1, 1, -1).expand(text_ids.shape[0], -1, -1)
            audio_emb = torch.cat([bos, audio_emb], dim=1)
        inputs_embeds = torch.cat([text_emb, audio_emb], dim=1)
        outputs = self.llama(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        audio_hidden = hidden[:, text_ids.shape[1]:, :]
        logits = self.lm_head(audio_hidden)
        return audio_hidden, logits
