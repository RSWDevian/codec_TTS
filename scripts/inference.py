#!/usr/bin/env python
"""Unified inference CLI: --backend csm (pretrained Sesame CSM-1B, needs HF
auth) or --backend hindi (our own from-scratch backbone+depth model).

Usage:
  python scripts/inference.py --backend csm --text "Hello from Sesame." --output outputs/audio/csm_test.wav
  python scripts/inference.py --backend hindi --text "नमस्ते" --output outputs/audio/hindi_test.wav
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


def build_hindi_pipeline(config_path: str, dataset_config_path: str, backbone_config_path: str, depth_config_path: str = "configs/backbone/hindi_depth_config.yaml", tokenizer_type: str = "hindi"):
    from inference.pipeline.hindi_pipeline import HindiTTSPipeline, TOKENIZER_CLASSES
    from models.backbone.transformer.config import HindiBackboneConfig
    from models.depth_transformer.config import HindiDepthConfig

    cfg = load_yaml_config(config_path)
    dataset_cfg = load_yaml_config(dataset_config_path)
    backbone_cfg_dict = load_yaml_config(backbone_config_path)
    depth_cfg_dict = load_yaml_config(depth_config_path)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints found in {checkpoint_dir} -- run training first"
        )
    checkpoint_path = checkpoints[-1]

    cls = TOKENIZER_CLASSES.get(tokenizer_type)
    if cls is None:
        raise ValueError(f"unknown tokenizer type: {tokenizer_type}, choose from {list(TOKENIZER_CLASSES)}")
    tokenizer = cls.load(dataset_cfg["vocab_path"])
    backbone_cfg = HindiBackboneConfig(text_vocab_size=tokenizer.vocab_size, **backbone_cfg_dict)
    depth_cfg = HindiDepthConfig(backbone_hidden_size=backbone_cfg.hidden_size, **depth_cfg_dict)

    pipeline = HindiTTSPipeline(
        checkpoint_path=str(checkpoint_path),
        vocab_path=dataset_cfg["vocab_path"],
        backbone_cfg=backbone_cfg,
        depth_cfg=depth_cfg,
        tokenizer_type=tokenizer_type,
        device=cfg["device"] if torch.cuda.is_available() else "cpu",
    )
    return pipeline, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["csm", "hindi"], default="hindi")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="0")
    parser.add_argument("--output", default="outputs/audio/output.wav")
    parser.add_argument("--tokenizer-type", choices=["grapheme", "hindi"], default=None)
    parser.add_argument("--csm-config", default="configs/inference/csm_default.yaml")
    parser.add_argument("--hindi-config", default="configs/inference/hindi_default.yaml")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--backbone-config", default="configs/backbone/hindi_backbone.yaml")
    parser.add_argument("--depth-config", default="configs/backbone/hindi_depth_config.yaml")
    parser.add_argument("--hindi-inference-config", default="configs/inference/hindi_default.yaml")
    args = parser.parse_args()

    # These two only make sense per-backend; default them off the chosen
    # backend rather than a single global default, since `hindi` silently
    # loading the ljspeech/grapheme tokenizer produces garbage without erroring.
    if args.backend == "hindi":
        args.dataset_config = args.dataset_config or "configs/datasets/hindi.yaml"
        args.tokenizer_type = args.tokenizer_type or "hindi"
    else:
        args.dataset_config = args.dataset_config or "configs/datasets/ljspeech.yaml"
        args.tokenizer_type = args.tokenizer_type or "grapheme"

    if args.backend == "csm":
        pipeline, cfg = build_csm_pipeline(args.csm_config)
        kwargs = {"speaker": args.speaker}
    else:
        pipeline, cfg = build_hindi_pipeline(args.hindi_config, args.dataset_config, args.backbone_config, args.depth_config, args.tokenizer_type)
        kwargs = {
            k: cfg[k]
            for k in ("frames_per_char", "max_frames", "top_k", "temperature", "repetition_penalty")
            if k in cfg
        }

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
