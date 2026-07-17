"""Cross-entropy losses for the joint backbone + depth transformer training step."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def backbone_loss(
    logits: torch.Tensor, target_codebook0: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """logits: (B, Ta, vocab) at position t predicts target_codebook0[:, t+1].
    target_mask: (B, Ta) 1 for valid (non-padding) audio frames.
    """
    pred_logits = logits[:, :-1, :]
    targets = target_codebook0[:, 1:]
    mask = target_mask[:, 1:].reshape(-1).bool()
    pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])[mask]
    targets = targets.reshape(-1)[mask]
    return F.cross_entropy(pred_logits, targets)


def depth_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits: (N, 7, vocab), targets: (N, 7), mask: (N,) 1 for valid frames."""
    mask = mask.bool()
    logits = logits[mask].reshape(-1, logits.shape[-1])
    targets = targets[mask].reshape(-1)
    return F.cross_entropy(logits, targets)
