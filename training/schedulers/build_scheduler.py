from __future__ import annotations

import torch
from transformers import get_constant_schedule_with_warmup


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int = 50):
    return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
