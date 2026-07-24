from __future__ import annotations

import torch
from transformers import get_cosine_schedule_with_warmup


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int = 500, num_training_steps: int = 200000):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
