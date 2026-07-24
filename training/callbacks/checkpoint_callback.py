from pathlib import Path

import torch


class CheckpointCallback:
    def __init__(self, checkpoint_dir: str, checkpoint_every: int = 1000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.best_val_loss = float("inf")

    def on_step(self, step: int, state: dict) -> None:
        if step == 0 or step % self.checkpoint_every != 0:
            return
        path = self.checkpoint_dir / f"step_{step}.pt"
        torch.save(state, path)
        print(f"Saved checkpoint -> {path}")

    def on_validation(self, step: int, val_loss: float, state: dict) -> None:
        if val_loss >= self.best_val_loss:
            return
        self.best_val_loss = val_loss
        path = self.checkpoint_dir / "best.pt"
        torch.save(state, path)
        print(f"New best val_loss={val_loss:.4f} (step {step}) -> saved {path}")
