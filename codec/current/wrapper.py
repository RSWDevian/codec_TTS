"""Thin wrapper around the pretrained Kyutai Mimi codec (transformers.MimiModel).

Used to tokenize/detokenize audio for the training scaffold in
training/ and inference/pipeline/hindi_pipeline.py. Not used by the pretrained
CSM pipeline (inference/pipeline/csm_pipeline.py), which bundles its own
Mimi decoding internally.
"""

from __future__ import annotations
import numpy as np
import torch
from transformers import AutoFeatureExtractor, MimiModel

MODEL_ID = "kyutai/mimi"
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
FRAME_RATE_HZ = 12.5
SAMPLE_RATE = 24000


class MimiCodec:
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.model = MimiModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    @torch.no_grad()
    def encode(self, waveform: np.ndarray, sample_rate: int) -> torch.LongTensor:
        """waveform: mono float32 array at `sample_rate` (must be SAMPLE_RATE).

        Returns LongTensor of shape (batch, NUM_CODEBOOKS, num_frames), values
        in [0, CODEBOOK_SIZE - 1], at FRAME_RATE_HZ.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE}Hz input, got {sample_rate}Hz")
        inputs = self.feature_extractor(
            raw_audio=waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        encoder_outputs = self.model.encode(inputs["input_values"], num_quantizers=NUM_CODEBOOKS)
        return encoder_outputs.audio_codes

    @torch.no_grad()
    def decode(self, codes: torch.LongTensor) -> np.ndarray:
        """codes: LongTensor (batch, NUM_CODEBOOKS, num_frames) -> mono waveform
        (num_samples,) @ SAMPLE_RATE. audio_values is (batch, channels, num_samples);
        Mimi is mono, so we drop both the batch and channel dims.
        """
        codes = codes.to(self.device)
        decoded = self.model.decode(codes)
        audio = decoded.audio_values[0, 0].to(torch.float32).cpu().numpy()
        return audio
