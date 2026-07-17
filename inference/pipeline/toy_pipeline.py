"""Toy inference backend: our own from-scratch backbone + depth transformer
(trained via scripts/train_backbone.py), decoded through Mimi
(codec/current/wrapper.py). No KV cache -- recomputes the full forward each
step, which is fine for the toy model's size but not representative of an
efficient runtime (see inference/kv_cache/ for that future work).

Generation seeds the audio sequence with a single zero-token "frame" before
any real prediction happens, since training never computes a loss for
predicting the very first frame's codebook-0 (it's only ever an input, see
training/losses/ce_loss.py:backbone_loss). This is a known toy-prototype
simplification -- a proper implementation would use a dedicated learned
audio-BOS embedding instead of token id 0.
"""

from __future__ import annotations

import numpy as np
import torch

from codec.current.wrapper import MimiCodec, SAMPLE_RATE
from inference.pipeline.base import BasePipeline
from models.backbone.transformer.config import ToyBackboneConfig
from models.backbone.transformer.model import ToyBackbone
from models.depth_transformer.config import ToyDepthConfig
from models.depth_transformer.model import ToyDepthTransformer
from models.embeddings.codec_embeddings import MimiCodebookEmbeddings
from preprocessing.text.tokenizer import GraphemeTokenizer


class ToyTTSPipeline(BasePipeline):
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        backbone_cfg: ToyBackboneConfig,
        depth_cfg: ToyDepthConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.tokenizer = GraphemeTokenizer.load(vocab_path)

        codec_embeddings = MimiCodebookEmbeddings(hidden_size=backbone_cfg.hidden_size)
        self.backbone = ToyBackbone(backbone_cfg, codec_embeddings).to(device)
        self.depth = ToyDepthTransformer(depth_cfg, codec_embeddings).to(device)

        state = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(state["backbone"])
        self.depth.load_state_dict(state["depth"])
        self.backbone.eval()
        self.depth.eval()

        self.codec = MimiCodec(device=device)

    @torch.no_grad()
    def synthesize(
        self, text: str, max_frames: int = 200, frames_per_char: float = 2.0, **kwargs
    ) -> tuple[np.ndarray, int]:
        text_ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)
        num_frames = min(max_frames, max(8, int(len(text) * frames_per_char)))

        codebook0 = torch.zeros((1, 1), dtype=torch.long, device=self.device)  # seed, see module docstring
        all_frame_codes = []

        for _ in range(num_frames):
            hidden, logits = self.backbone(text_ids, codebook0)
            next_codebook0 = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            last_hidden = hidden[:, -1, :]

            frame_codes = [next_codebook0.squeeze(1)]
            depth_input = torch.zeros((1, 6), dtype=torch.long, device=self.device)
            for i in range(7):
                depth_logits = self.depth(last_hidden, depth_input)
                tok = depth_logits[:, i, :].argmax(dim=-1)
                frame_codes.append(tok)
                if i < 6:
                    depth_input[:, i] = tok

            all_frame_codes.append(torch.stack(frame_codes, dim=1))  # (1, 8)
            codebook0 = torch.cat([codebook0, next_codebook0], dim=1)

        codes = torch.stack(all_frame_codes, dim=2)  # (1, 8, num_frames)
        waveform = self.codec.decode(codes)
        return waveform, SAMPLE_RATE
