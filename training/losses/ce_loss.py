import torch
import torch.nn.functional as F


def backbone_loss(
    logits: torch.Tensor, target_codebook0: torch.Tensor, target_mask: torch.Tensor, add_bos: bool = True
) -> torch.Tensor:
    if add_bos:
        # target_codebook0/target_mask already include the trailing <audio_eos>
        # position, so logits (one per bos/audio-token consumed) line up 1:1
        # with targets -- no slicing needed.
        pred_logits = logits
        targets = target_codebook0
        mask = target_mask
    else:
        pred_logits = logits[:, :-1, :]
        targets = target_codebook0[:, 1:]
        mask = target_mask[:, 1:]
    mask = mask.reshape(-1).bool()
    pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])[mask]
    targets = targets.reshape(-1)[mask]
    return F.cross_entropy(pred_logits, targets)


def depth_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    logits = logits[mask].reshape(-1, logits.shape[-1])
    targets = targets[mask].reshape(-1)
    return F.cross_entropy(logits, targets)
