"""Dataset over Mimi-tokenized LJSpeech utterances (see scripts/tokenize_dataset.py)."""

from __future__ import annotations

import csv

import torch
from torch.utils.data import Dataset


class ToyTTSDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.token_paths = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.token_paths.append(row["token_path"])

    def __len__(self) -> int:
        return len(self.token_paths)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.token_paths[idx])
        return {
            "text_ids": torch.tensor(data["text_ids"], dtype=torch.long),
            "codes": data["codes"].long(),  # (8, T)
        }


def collate_fn(batch: list[dict], pad_id: int) -> dict:
    max_t_text = max(item["text_ids"].shape[0] for item in batch)
    max_t_audio = max(item["codes"].shape[1] for item in batch)
    b = len(batch)

    text_ids = torch.full((b, max_t_text), pad_id, dtype=torch.long)
    text_mask = torch.zeros((b, max_t_text), dtype=torch.long)
    codes = torch.zeros((b, 8, max_t_audio), dtype=torch.long)
    audio_mask = torch.zeros((b, max_t_audio), dtype=torch.long)

    for i, item in enumerate(batch):
        t = item["text_ids"].shape[0]
        text_ids[i, :t] = item["text_ids"]
        text_mask[i, :t] = 1
        a = item["codes"].shape[1]
        codes[i, :, :a] = item["codes"]
        audio_mask[i, :a] = 1

    return {"text_ids": text_ids, "text_mask": text_mask, "codes": codes, "audio_mask": audio_mask}
