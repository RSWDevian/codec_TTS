#!/usr/bin/env python
"""Tokenize a Hindi dataset into Mimi codec codes for training.

Usage:
  python scripts/tokenize_dataset.py --config configs/datasets/hindi.yaml

Expects data in:
  data/raw/hindi/*.wav  +  *.txt  (Dewanagari Hindi text)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from codec.current.wrapper import MimiCodec, SAMPLE_RATE
from preprocessing.text.hindi_tokenizer import HindiTokenizer


def resample_and_normalize(waveform, orig_sr, target_sr):
    if orig_sr != target_sr:
        import torchaudio.functional as F
        waveform = F.resample(torch.from_numpy(waveform), orig_sr, target_sr).numpy()
    peak = max(abs(waveform.max()), abs(waveform.min()))
    if peak > 0:
        waveform = waveform / peak
    return waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/hindi.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Inline loader to avoid name conflict with pip `datasets` package
    root_dir = Path(cfg["root_dir"])
    all_ids = sorted(
        w.stem for w in root_dir.glob("*.wav")
        if w.with_suffix(".txt").exists()
    )
    random.seed(42)
    random.shuffle(all_ids)

    texts = {}
    for utt_id in all_ids:
        txt_path = root_dir / f"{utt_id}.txt"
        texts[utt_id] = txt_path.read_text(encoding="utf-8").strip()
    tokenizer = HindiTokenizer.build_from_texts(list(texts.values()))
    Path(cfg["vocab_path"]).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg["vocab_path"])
    print(f"Vocab size: {tokenizer.vocab_size}")

    codec = MimiCodec(device=args.device)

    tokenized_dir = Path(cfg["tokenized_dir"])
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for utt_id in tqdm(all_ids, desc="tokenizing"):
        audio_path = root_dir / f"{utt_id}.wav"
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        if sr != cfg["target_sample_rate"]:
            waveform = resample_and_normalize(waveform, sr, cfg["target_sample_rate"])
        waveform = waveform.astype("float32")
        codes = codec.encode(waveform, cfg["target_sample_rate"])
        text = texts[utt_id]
        text_ids = tokenizer.encode(text)
        out_path = tokenized_dir / f"{utt_id}.pt"
        torch.save({"codes": codes[0], "text_ids": text_ids, "text": text}, out_path)
        manifest_rows.append([utt_id, text, str(out_path), codes.shape[-1]])

    with open(cfg["manifest_path"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "text", "token_path", "num_frames"])
        writer.writerows(manifest_rows)

    print(f"Tokenized {len(all_ids)} utterances -> {tokenized_dir}")
    print(f"Manifest -> {cfg['manifest_path']}")
    print(f"Vocab -> {cfg['vocab_path']}")


if __name__ == "__main__":
    main()
