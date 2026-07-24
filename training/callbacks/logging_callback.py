import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class LoggingCallback:
    def __init__(self, log_dir: str, log_every: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.writer = SummaryWriter(str(self.log_dir))

    def on_step(self, step: int, losses: dict) -> None:
        if step % self.log_every != 0:
            return
        record = {"step": step, **losses}
        print(record)
        with open(self.log_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        for k, v in losses.items():
            self.writer.add_scalar(k, v, step)

    def on_val(self, step: int, losses: dict) -> None:
        record = {"step": step, "split": "val", **{f"val_{k}": v for k, v in losses.items()}}
        print(record)
        with open(self.log_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        for k, v in losses.items():
            self.writer.add_scalar(f"val/{k}", v, step)
