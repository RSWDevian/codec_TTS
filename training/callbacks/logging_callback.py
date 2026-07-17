from __future__ import annotations

import json
from pathlib import Path


class LoggingCallback:
    def __init__(self, log_path: str, log_every: int = 50):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

    def on_step(self, step: int, losses: dict) -> None:
        if step % self.log_every != 0:
            return
        record = {"step": step, **losses}
        print(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
