# training

Current: `backbone/`, `losses/`, `optimizers/`, `schedulers/`, `callbacks/`, and `trainer.py` implement the joint toy backbone+depth overfit run (see `scripts/train_backbone.py`). `depth/` is a stub today — see `training/depth/README.md`.

Future: `codec/` will hold configs/entrypoints for training a custom codec (the actual codec architecture lives in `codec/future/trainer/`). `finetune/` is reserved for fine-tuning a trained backbone/depth model on new voices or domains.
