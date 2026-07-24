"""Pretrained inference backend: Sesame CSM-1B (sesame/csm-1b) via
transformers.CsmForConditionalGeneration.

Mimi decoding happens *internally* in this checkpoint (it bundles its own
Mimi weights) -- unlike the hindi backend (hindi_pipeline.py), this does NOT
go through codec/current/wrapper.py.

Requires HF auth + accepted license terms for sesame/csm-1b and
meta-llama/Llama-3.2-1B (see scripts/setup.sh for instructions).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration

from inference.pipeline.base import BasePipeline

MODEL_ID = "sesame/csm-1b"
SAMPLE_RATE = 24000


class CSMPipeline(BasePipeline):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = CsmForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map=device, torch_dtype=dtype
        )

    def synthesize(self, text: str, speaker: str = "0", **kwargs) -> tuple[np.ndarray, int]:
        conversation = [{"role": speaker, "content": [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to(self.device)
        audio = self.model.generate(**inputs, output_audio=True, **kwargs)
        waveform = audio[0].to(torch.float32).cpu().numpy()
        return waveform, SAMPLE_RATE
