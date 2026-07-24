from __future__ import annotations

from transformers import LlamaConfig


class HindiDepthConfig:
    def __init__(
        self,
        backbone_hidden_size: int,
        audio_vocab_size: int = 2048,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 256,
        num_depth_codebooks: int = 7,
    ):
        self.backbone_hidden_size = backbone_hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_depth_codebooks = num_depth_codebooks

    def to_llama_config(self) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=self.audio_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.num_depth_codebooks + 1,
        )
