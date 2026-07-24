#!/usr/bin/env python
"""Hindi TTS training frontend.

Usage:
  python scripts/train.py --config configs/training/hindi_training.yaml
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from models.backbone.transformer.config import HindiBackboneConfig
from models.backbone.transformer.model import HindiBackbone
from models.common.model_utils import count_parameters, load_yaml_config
from models.depth_transformer.config import HindiDepthConfig
from models.depth_transformer.model import HindiDepthTransformer
from models.embeddings.codec_embeddings import AUDIO_EOS_ID, MimiCodebookEmbeddings
from preprocessing.text.hindi_tokenizer import HindiTokenizer
from training.backbone.dataset import TTSTokenDataset, collate_fn
from training.callbacks.checkpoint_callback import CheckpointCallback
from training.callbacks.logging_callback import LoggingCallback
from training.optimizers.build_optimizer import build_optimizer
from training.schedulers.build_scheduler import build_scheduler
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/hindi_training.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets/hindi.yaml")
    parser.add_argument("--backbone-config", default="configs/backbone/hindi_backbone.yaml")
    parser.add_argument("--depth-config", default="configs/backbone/hindi_depth_config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    train_cfg = load_yaml_config(args.config)
    dataset_cfg = load_yaml_config(args.dataset_config)
    backbone_cfg_dict = load_yaml_config(args.backbone_config)
    depth_cfg_dict = load_yaml_config(args.depth_config)

    torch.manual_seed(train_cfg["seed"])

    tokenizer = HindiTokenizer.load(dataset_cfg["vocab_path"])

    val_fraction = train_cfg.get("val_fraction", 0.03)
    collate = functools.partial(collate_fn, pad_id=tokenizer.pad_id, eos_id=AUDIO_EOS_ID)

    dataset = TTSTokenDataset(dataset_cfg["manifest_path"], split="train", val_fraction=val_fraction)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = TTSTokenDataset(dataset_cfg["manifest_path"], split="val", val_fraction=val_fraction)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    codec_embeddings = MimiCodebookEmbeddings(hidden_size=backbone_cfg_dict["hidden_size"])

    backbone_cfg = HindiBackboneConfig(text_vocab_size=tokenizer.vocab_size, **backbone_cfg_dict)
    backbone = HindiBackbone(backbone_cfg, codec_embeddings)

    depth_cfg = HindiDepthConfig(backbone_hidden_size=backbone_cfg.hidden_size, **depth_cfg_dict)
    depth = HindiDepthTransformer(depth_cfg, codec_embeddings)

    optimizer = build_optimizer([backbone, depth], lr=train_cfg["lr"])
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        num_training_steps=train_cfg["num_steps"] // train_cfg["gradient_accumulation_steps"],
    )

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        backbone.load_state_dict(state["backbone"])
        depth.load_state_dict(state["depth"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_step = state.get("step", 0) + 1
        print(f"Resumed from step {start_step}")
        train_cfg["num_steps"] = train_cfg["num_steps"] - start_step

    print(f"Backbone params: {count_parameters(backbone):,}")
    print(f"Depth params: {count_parameters(depth):,}")
    print(f"Total params: {count_parameters(backbone) + count_parameters(depth):,}")

    log_dir = f"outputs/logs/hindi_train_{int(time.time())}"
    ckpt_dir = "checkpoints/hindi"

    logging_callback = LoggingCallback(log_dir, log_every=train_cfg["log_every"])
    checkpoint_callback = CheckpointCallback(ckpt_dir, checkpoint_every=train_cfg["checkpoint_every"])

    trainer = Trainer(
        backbone,
        depth,
        optimizer,
        scheduler,
        args.device,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        logging_callback=logging_callback,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(
        dataloader,
        num_steps=train_cfg["num_steps"],
        start_step=start_step,
        val_dataloader=val_dataloader,
        val_every=train_cfg.get("val_every"),
    )

    print("Training complete.")
    print(f"Logs: {log_dir}/")
    print(f"Checkpoints: {ckpt_dir}/")
    print(f"Launch tensorboard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
