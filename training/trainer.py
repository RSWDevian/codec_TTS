"""Joint training loop for the toy backbone + depth transformer.

Per step: backbone consumes [text, codebook-0 frames] and produces, at each
audio-frame position, a hidden state used for two things -- (1) predicting
the *next* frame's codebook-0 token, and (2) conditioning the depth
transformer to predict *that same* frame's codebooks 1-7. This mirrors the
Moshi/CSM design where one backbone hidden state serves both roles.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models.backbone.transformer.model import ToyBackbone
from models.depth_transformer.model import ToyDepthTransformer
from training.losses.ce_loss import backbone_loss, depth_loss


class ToyTrainer:
    def __init__(
        self,
        backbone: ToyBackbone,
        depth: ToyDepthTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        logging_callback=None,
        checkpoint_callback=None,
    ):
        self.backbone = backbone.to(device)
        self.depth = depth.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logging_callback = logging_callback
        self.checkpoint_callback = checkpoint_callback

    def step(self, batch: dict) -> dict:
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        codes = batch["codes"].to(self.device)
        audio_mask = batch["audio_mask"].to(self.device)

        b, _, ta = codes.shape
        attention_mask = torch.cat([text_mask, audio_mask], dim=1)

        hidden, logits0 = self.backbone(text_ids, codes[:, 0, :], attention_mask=attention_mask)
        loss_bb = backbone_loss(logits0, codes[:, 0, :], audio_mask)

        flat_hidden = hidden.reshape(b * ta, -1)
        flat_targets = codes[:, 1:8, :].permute(0, 2, 1).reshape(b * ta, 7)
        flat_input_codes = codes[:, 1:7, :].permute(0, 2, 1).reshape(b * ta, 6)
        flat_mask = audio_mask.reshape(b * ta)

        depth_logits = self.depth(flat_hidden, flat_input_codes)
        loss_depth = depth_loss(depth_logits, flat_targets, flat_mask)

        loss = loss_bb + loss_depth

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "loss_backbone": loss_bb.item(),
            "loss_depth": loss_depth.item(),
        }

    def state_dict(self, step: int) -> dict:
        return {"backbone": self.backbone.state_dict(), "depth": self.depth.state_dict(), "step": step}

    def fit(self, dataloader: DataLoader, num_steps: int) -> None:
        step = 0
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break
                losses = self.step(batch)
                if self.logging_callback:
                    self.logging_callback.on_step(step, losses)
                if self.checkpoint_callback:
                    self.checkpoint_callback.on_step(step, self.state_dict(step))
                step += 1
        if self.checkpoint_callback:
            final_path = self.checkpoint_callback.checkpoint_dir / f"step_{step}_final.pt"
            torch.save(self.state_dict(step), final_path)
            print(f"Saved final checkpoint -> {final_path}")
