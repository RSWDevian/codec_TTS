from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from models.backbone.transformer.model import HindiBackbone
from models.depth_transformer.model import HindiDepthTransformer
from training.losses.ce_loss import backbone_loss, depth_loss


class Trainer:
    def __init__(
        self,
        backbone: HindiBackbone,
        depth: HindiDepthTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        gradient_accumulation_steps: int = 1,
        logging_callback=None,
        checkpoint_callback=None,
    ):
        self.backbone = backbone.to(device)
        self.depth = depth.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_callback = logging_callback
        self.checkpoint_callback = checkpoint_callback

        use_bf16 = device == "cuda"
        self.amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
            self.backbone.llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.depth.llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def step(self, batch: dict) -> dict:
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        codes = batch["codes"].to(self.device)
        audio_mask = batch["audio_mask"].to(self.device)
        codebook0_ext = batch["codebook0_ext"].to(self.device)
        audio_mask_ext = batch["audio_mask_ext"].to(self.device)

        b, _, ta = codes.shape
        bos_mask = torch.ones((b, 1), dtype=torch.long, device=self.device)
        attention_mask = torch.cat([text_mask, bos_mask, audio_mask], dim=1)

        with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype):
            hidden, logits0 = self.backbone(text_ids, codes[:, 0, :], attention_mask=attention_mask, add_bos=True)
            loss_bb = backbone_loss(logits0, codebook0_ext, audio_mask_ext, add_bos=True)

            audio_hidden = hidden[:, 1:, :]
            flat_hidden = audio_hidden.reshape(b * ta, -1)
            flat_targets = codes[:, 1:8, :].permute(0, 2, 1).reshape(b * ta, 7)
            flat_input_codes = codes[:, 1:7, :].permute(0, 2, 1).reshape(b * ta, 6)
            flat_mask = audio_mask.reshape(b * ta)

            depth_logits = self.depth(flat_hidden, flat_input_codes)
            loss_depth = depth_loss(depth_logits, flat_targets, flat_mask)

            loss = loss_bb + loss_depth
        return {"loss": loss, "loss_backbone": loss_bb, "loss_depth": loss_depth}

    def state_dict(self, step: int) -> dict:
        return {
            "backbone": self.backbone.state_dict(),
            "depth": self.depth.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.backbone.eval()
        self.depth.eval()
        totals = {"loss": 0.0, "loss_backbone": 0.0, "loss_depth": 0.0}
        n = 0
        for batch in dataloader:
            losses = self.step(batch)
            for k in totals:
                totals[k] += losses[k].item()
            n += 1
        self.backbone.train()
        self.depth.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    def fit(
        self,
        dataloader: DataLoader,
        num_steps: int,
        start_step: int = 0,
        val_dataloader: DataLoader | None = None,
        val_every: int | None = None,
    ) -> None:
        step = start_step
        target = start_step + num_steps
        self.backbone.train()
        self.depth.train()
        while step < target:
            for batch in dataloader:
                if step >= target:
                    break

                losses = self.step(batch)
                loss = losses["loss"]
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.depth.parameters()), 1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                log_losses = {k: v.item() for k, v in losses.items()}
                if self.logging_callback:
                    self.logging_callback.on_step(step, log_losses)
                if self.checkpoint_callback:
                    self.checkpoint_callback.on_step(step, self.state_dict(step))
                step += 1

                if val_dataloader is not None and val_every and step % val_every == 0:
                    val_losses = self.evaluate(val_dataloader)
                    if self.logging_callback:
                        self.logging_callback.on_val(step, val_losses)
                    if self.checkpoint_callback:
                        self.checkpoint_callback.on_validation(step, val_losses["loss"], self.state_dict(step))

        if self.checkpoint_callback:
            final_path = self.checkpoint_callback.checkpoint_dir / f"step_{step}_final.pt"
            torch.save(self.state_dict(step), final_path)
            print(f"Saved final checkpoint -> {final_path}")
