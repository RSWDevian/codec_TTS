#!/usr/bin/env python
"""Tokenizes a subset of LJSpeech into Mimi codec codes for toy training.

Usage: python scripts/tokenize_dataset.py --config configs/datasets/ljspeech.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from tqdm import tqdm

from codec.current.wrapper import MimiCodec
from datasets.ljspeech.loader import LJSpeechLoader
from preprocessing.pipelines.ljspeech_tokenize_pipeline import tokenize_utterance
from preprocessing.text.normalize import normalize_text
from preprocessing.text.tokenizer import GraphemeTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/ljspeech.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    loader = LJSpeechLoader(cfg["root_dir"])
    all_ids = loader.list_utterances()
    random.seed(cfg["seed"])
    random.shuffle(all_ids)
    subset_ids = all_ids[: cfg["subset_size"]]

    texts = {utt_id: normalize_text(loader.get_text(utt_id)) for utt_id in subset_ids}
    tokenizer = GraphemeTokenizer.build_from_texts(list(texts.values()))
    Path(cfg["vocab_path"]).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg["vocab_path"])

    codec = MimiCodec(device=args.device)

    tokenized_dir = Path(cfg["tokenized_dir"])
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for utt_id in tqdm(subset_ids, desc="tokenizing"):
        audio_path = loader.get_audio_path(utt_id)
        codes = tokenize_utterance(
            codec, str(audio_path), cfg["native_sample_rate"], cfg["target_sample_rate"]
        )
        text = texts[utt_id]
        text_ids = tokenizer.encode(text)
        out_path = tokenized_dir / f"{utt_id}.pt"
        torch.save({"codes": codes, "text_ids": text_ids, "text": text}, out_path)
        manifest_rows.append([utt_id, text, str(out_path), codes.shape[-1]])

    with open(cfg["manifest_path"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "text", "token_path", "num_frames"])
        writer.writerows(manifest_rows)

    print(f"Tokenized {len(subset_ids)} utterances -> {tokenized_dir}")
    print(f"Manifest -> {cfg['manifest_path']}")
    print(f"Text vocab ({tokenizer.vocab_size} chars) -> {cfg['vocab_path']}")


if __name__ == "__main__":
    main()
