#!/usr/bin/env python
"""Toy overfit training run: joint backbone + depth transformer on a tiny
LJSpeech subset, to prove the from-scratch training pipeline works end to
end. Expect training loss to drop sharply (near-memorization); audio
quality from the resulting checkpoint is expected to be poor/babble -- the
bar is a non-silent, non-NaN wav out (see inference/pipeline/toy_pipeline.py).

Usage: python scripts/train_backbone.py --config configs/training/toy_overfit.yaml
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from models.backbone.transformer.config import ToyBackboneConfig
from models.backbone.transformer.model import ToyBackbone
from models.common.model_utils import count_parameters, load_yaml_config
from models.depth_transformer.config import ToyDepthConfig
from models.depth_transformer.model import ToyDepthTransformer
from models.embeddings.codec_embeddings import MimiCodebookEmbeddings
from preprocessing.text.tokenizer import GraphemeTokenizer
from training.backbone.dataset import ToyTTSDataset, collate_fn
from training.callbacks.checkpoint_callback import CheckpointCallback
from training.callbacks.logging_callback import LoggingCallback
from training.optimizers.build_optimizer import build_optimizer
from training.schedulers.build_scheduler import build_scheduler
from training.trainer import ToyTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/toy_overfit.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets/ljspeech.yaml")
    parser.add_argument("--backbone-config", default="configs/backbone/toy_backbone.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_cfg = load_yaml_config(args.config)
    dataset_cfg = load_yaml_config(args.dataset_config)
    backbone_cfg_dict = load_yaml_config(args.backbone_config)

    torch.manual_seed(train_cfg["seed"])

    tokenizer = GraphemeTokenizer.load(dataset_cfg["vocab_path"])

    dataset = ToyTTSDataset(dataset_cfg["manifest_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=functools.partial(collate_fn, pad_id=tokenizer.pad_id),
    )

    codec_embeddings = MimiCodebookEmbeddings(hidden_size=backbone_cfg_dict["hidden_size"])

    backbone_cfg = ToyBackboneConfig(text_vocab_size=tokenizer.vocab_size, **backbone_cfg_dict)
    backbone = ToyBackbone(backbone_cfg, codec_embeddings)

    depth_cfg = ToyDepthConfig(backbone_hidden_size=backbone_cfg.hidden_size)
    depth = ToyDepthTransformer(depth_cfg, codec_embeddings)

    print(f"backbone params: {count_parameters(backbone):,}")
    print(f"depth params: {count_parameters(depth):,}")

    optimizer = build_optimizer([backbone, depth], lr=train_cfg["lr"])
    scheduler = build_scheduler(optimizer, warmup_steps=train_cfg["warmup_steps"])

    logging_callback = LoggingCallback("outputs/logs/toy_overfit.jsonl", log_every=train_cfg["log_every"])
    checkpoint_callback = CheckpointCallback(
        "checkpoints/toy_overfit", checkpoint_every=train_cfg["checkpoint_every"]
    )

    trainer = ToyTrainer(
        backbone, depth, optimizer, scheduler, args.device, logging_callback, checkpoint_callback
    )
    trainer.fit(dataloader, num_steps=train_cfg["num_steps"])


if __name__ == "__main__":
    main()
