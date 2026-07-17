from __future__ import annotations

from pathlib import Path

import torch


class CheckpointCallback:
    def __init__(self, checkpoint_dir: str, checkpoint_every: int = 500):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every

    def on_step(self, step: int, state: dict) -> None:
        if step == 0 or step % self.checkpoint_every != 0:
            return
        path = self.checkpoint_dir / f"step_{step}.pt"
        torch.save(state, path)
        print(f"Saved checkpoint -> {path}")
