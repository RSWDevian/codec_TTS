#!/usr/bin/env python
"""Unified inference CLI: --backend csm (pretrained Sesame CSM-1B, needs HF
auth) or --backend toy (our own from-scratch backbone+depth model).

Usage:
  python scripts/inference.py --backend csm --text "Hello from Sesame." --output outputs/audio/csm_test.wav
  python scripts/inference.py --backend toy --text "hello there" --output outputs/audio/toy_test.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch

from models.common.model_utils import load_yaml_config


def build_csm_pipeline(config_path: str):
    from inference.pipeline.csm_pipeline import CSMPipeline

    cfg = load_yaml_config(config_path)
    dtype = getattr(torch, cfg["dtype"])
    return CSMPipeline(device=cfg["device"], dtype=dtype), cfg


def build_toy_pipeline(config_path: str, dataset_config_path: str, backbone_config_path: str):
    from inference.pipeline.toy_pipeline import ToyTTSPipeline
    from models.backbone.transformer.config import ToyBackboneConfig
    from models.depth_transformer.config import ToyDepthConfig
    from preprocessing.text.tokenizer import GraphemeTokenizer

    cfg = load_yaml_config(config_path)
    dataset_cfg = load_yaml_config(dataset_config_path)
    backbone_cfg_dict = load_yaml_config(backbone_config_path)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints found in {checkpoint_dir} -- run scripts/train_backbone.py first"
        )
    checkpoint_path = checkpoints[-1]

    tokenizer = GraphemeTokenizer.load(dataset_cfg["vocab_path"])
    backbone_cfg = ToyBackboneConfig(text_vocab_size=tokenizer.vocab_size, **backbone_cfg_dict)
    depth_cfg = ToyDepthConfig(backbone_hidden_size=backbone_cfg.hidden_size)

    pipeline = ToyTTSPipeline(
        checkpoint_path=str(checkpoint_path),
        vocab_path=dataset_cfg["vocab_path"],
        backbone_cfg=backbone_cfg,
        depth_cfg=depth_cfg,
        device=cfg["device"] if torch.cuda.is_available() else "cpu",
    )
    return pipeline, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["csm", "toy"], default="toy")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="0")
    parser.add_argument("--output", default="outputs/audio/output.wav")
    parser.add_argument("--csm-config", default="configs/inference/csm_default.yaml")
    parser.add_argument("--toy-config", default="configs/inference/toy_default.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets/ljspeech.yaml")
    parser.add_argument("--backbone-config", default="configs/backbone/toy_backbone.yaml")
    args = parser.parse_args()

    if args.backend == "csm":
        pipeline, cfg = build_csm_pipeline(args.csm_config)
        kwargs = {"speaker": args.speaker}
    else:
        pipeline, cfg = build_toy_pipeline(args.toy_config, args.dataset_config, args.backbone_config)
        kwargs = {}

    start = time.time()
    waveform, sample_rate = pipeline.synthesize(args.text, **kwargs)
    elapsed = time.time() - start

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, sample_rate)

    duration = len(waveform) / sample_rate
    print(f"backend={args.backend} latency={elapsed:.2f}s audio_duration={duration:.2f}s -> {output_path}")


if __name__ == "__main__":
    main()
