#!/usr/bin/env python
"""Extract IISc dataset from HF cache and save as .wav + .txt files.

Reads cached .arrow files directly with pyarrow, extracts WAV bytes and text.

Usage:
  python scripts/prepare_iisc_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
from tqdm import tqdm


def read_arrow_table(path: Path) -> pa.Table:
    with open(path, "rb") as f:
        magic = f.read(4)
    # Arrow IPC files start with "ARROW1", streams start with continuation marker
    if magic == b"ARROW":
        with open(path, "rb") as f:
            reader = pa.ipc.open_file(f)
            return reader.read_all()
    else:
        with open(path, "rb") as f:
            reader = pa.ipc.open_stream(f)
            return reader.read_all()


def main():
    root = Path("data/raw/hindi_iisc")
    root.mkdir(parents=True, exist_ok=True)

    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / \
        "somu9___iisc_mono_hindi_female" / "default" / "0.0.0"
    version_dirs = sorted(cache_root.iterdir())
    if not version_dirs:
        print("Cached dataset not found. Download it via HF datasets first.")
        sys.exit(1)

    arrow_dir = version_dirs[-1]
    train_files = sorted(arrow_dir.glob("*train*.arrow"))
    print(f"Found {len(train_files)} train arrow files")

    utt_idx = 0
    for arrow_path in tqdm(train_files, desc="Extracting"):
        table = read_arrow_table(arrow_path)
        texts = table.column("text").to_pylist()
        audio_arr = table.column("audio").to_pylist()

        for text, audio_entry in zip(texts, audio_arr):
            audio_entry = audio_entry or {}
            wav_bytes = audio_entry.get("bytes")
            if wav_bytes is None:
                continue
            wav_path = root / f"iisc_{utt_idx:06d}.wav"
            txt_path = root / f"iisc_{utt_idx:06d}.txt"
            wav_path.write_bytes(wav_bytes)
            txt_path.write_text((text or "").strip(), encoding="utf-8")
            utt_idx += 1

    print(f"Done. Saved {utt_idx} wav+txt pairs to {root}/")


if __name__ == "__main__":
    main()
