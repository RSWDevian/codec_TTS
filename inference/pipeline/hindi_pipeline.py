from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from codec.current.wrapper import MimiCodec, SAMPLE_RATE
from inference.pipeline.base import BasePipeline
from models.backbone.transformer.config import HindiBackboneConfig
from models.backbone.transformer.model import HindiBackbone
from models.depth_transformer.config import HindiDepthConfig
from models.depth_transformer.model import HindiDepthTransformer
from models.embeddings.codec_embeddings import AUDIO_EOS_ID, MimiCodebookEmbeddings
from preprocessing.text.hindi_tokenizer import HindiTokenizer
from preprocessing.text.tokenizer import GraphemeTokenizer

TOKENIZER_CLASSES = {"grapheme": GraphemeTokenizer, "hindi": HindiTokenizer}


@torch.no_grad()
def topk_sample(
    logits: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    previous_ids: list[int] | None = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if repetition_penalty != 1.0 and previous_ids:
        for tok in set(previous_ids):
            logits[..., tok] = torch.where(
                logits[..., tok] > 0,
                logits[..., tok] / repetition_penalty,
                logits[..., tok] * repetition_penalty,
            )
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    idx = torch.multinomial(topk_probs, 1)
    return topk_indices.gather(-1, idx).squeeze(-1)


class HindiTTSPipeline(BasePipeline):
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        backbone_cfg: HindiBackboneConfig,
        depth_cfg: HindiDepthConfig,
        tokenizer_type: str = "grapheme",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        cls = TOKENIZER_CLASSES.get(tokenizer_type, GraphemeTokenizer)
        self.tokenizer = cls.load(vocab_path)

        codec_embeddings = MimiCodebookEmbeddings(hidden_size=backbone_cfg.hidden_size)
        self.backbone = HindiBackbone(backbone_cfg, codec_embeddings).to(device)
        self.depth = HindiDepthTransformer(depth_cfg, codec_embeddings).to(device)

        state = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(state["backbone"])
        self.depth.load_state_dict(state["depth"])
        self.backbone.eval()
        self.depth.eval()

        self.codec = MimiCodec(device=device)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        max_frames: int = 400,
        frames_per_char: float = 1.65,
        top_k: int = 10,
        temperature: float = 0.8,
        repetition_penalty: float = 1.3,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        text_ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)
        # frames_per_char is now just a safety cap -- real duration is decided
        # by the model sampling <audio_eos> on codebook 0 below.
        num_frames = min(max_frames, max(8, int(len(text) * frames_per_char)))

        all_frame_codes: list[list[int]] = []

        for _ in range(num_frames):
            codebook0_ids = torch.tensor(
                [[c[0] for c in all_frame_codes]], dtype=torch.long, device=self.device
            ) if all_frame_codes else torch.zeros((1, 0), dtype=torch.long, device=self.device)

            attn_len = text_ids.shape[1] + 1 + codebook0_ids.shape[1]
            attn_mask = torch.ones((1, attn_len), dtype=torch.long, device=self.device)

            hidden, logits = self.backbone(
                text_ids, codebook0_ids, attention_mask=attn_mask, add_bos=True
            )

            next_logits = logits[:, -1, :]
            recent_ids = [c[0] for c in all_frame_codes[-20:]]
            next_codebook0 = topk_sample(
                next_logits, k=top_k, temperature=temperature,
                repetition_penalty=repetition_penalty, previous_ids=recent_ids,
            )

            if next_codebook0.item() == AUDIO_EOS_ID:
                break

            last_hidden = hidden[:, -1, :]
            frame_codes = [next_codebook0.item()]

            depth_input = torch.zeros((1, 6), dtype=torch.long, device=self.device)
            for i in range(7):
                depth_logits = self.depth(last_hidden, depth_input)
                tok = topk_sample(depth_logits[:, i, :], k=top_k, temperature=temperature)
                frame_codes.append(tok.item())
                if i < 6:
                    depth_input[:, i] = tok

            all_frame_codes.append(frame_codes)

        if not all_frame_codes:
            return np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE

        codes = torch.tensor([all_frame_codes], device=self.device).permute(0, 2, 1)
        waveform = self.codec.decode(codes)
        return waveform, SAMPLE_RATE
