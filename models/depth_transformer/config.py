from __future__ import annotations

from dataclasses import dataclass

from transformers import LlamaConfig


@dataclass
class ToyDepthConfig:
    backbone_hidden_size: int
    audio_vocab_size: int = 2048
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 256
    num_depth_codebooks: int = 7  # codebooks 1..7 (codebook 0 handled by the backbone)

    def to_llama_config(self) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=self.audio_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.num_depth_codebooks + 1,
        )
