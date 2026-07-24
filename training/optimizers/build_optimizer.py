"""AdamW builder with weight-decay filter and parameter deduplication."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(
    models: Iterable[torch.nn.Module], lr: float, weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    seen: set[int] = set()
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
