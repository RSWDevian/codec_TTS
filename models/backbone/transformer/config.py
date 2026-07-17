from __future__ import annotations
from dataclasses import dataclass
from transformers import LlamaConfig


@dataclass
class ToyBackboneConfig:
    text_vocab_size: int
    audio_vocab_size: int = 2048
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 512

    def to_llama_config(self) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=self.audio_vocab_size,  # unused directly, we pass inputs_embeds
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
        )
