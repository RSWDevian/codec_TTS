Depth transformer training is not decoupled today — it's trained jointly with the backbone in a single step inside `training/trainer.py` / `scripts/train_backbone.py`.

`scripts/train_depth.py` is wired (accepts `--checkpoint --freeze-backbone`, reuses `training/trainer.py`) for future decoupled depth fine-tuning, but is not exercised in today's prototype.
