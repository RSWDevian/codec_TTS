"""Dataset over Mimi-tokenized Hindi utterances for backbone + depth training."""

from __future__ import annotations

import csv
import random

import torch
from torch.utils.data import Dataset


class TTSTokenDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        val_fraction: float = 0.03,
        seed: int = 42,
    ):
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            all_paths = [row["token_path"] for row in reader]

        indices = list(range(len(all_paths)))
        random.Random(seed).shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx = set(indices[:n_val])

        if split == "val":
            self.token_paths = [all_paths[i] for i in sorted(val_idx)]
        else:
            self.token_paths = [all_paths[i] for i in range(len(all_paths)) if i not in val_idx]

    def __len__(self) -> int:
        return len(self.token_paths)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.token_paths[idx])
        return {
            "text_ids": torch.tensor(data["text_ids"], dtype=torch.long),
            "codes": data["codes"].long(),
        }


def collate_fn(batch: list[dict], pad_id: int, eos_id: int) -> dict:
    max_t_text = max(item["text_ids"].shape[0] for item in batch)
    max_t_audio = max(item["codes"].shape[1] for item in batch)
    b = len(batch)

    text_ids = torch.full((b, max_t_text), pad_id, dtype=torch.long)
    text_mask = torch.zeros((b, max_t_text), dtype=torch.long)
    codes = torch.zeros((b, 8, max_t_audio), dtype=torch.long)
    audio_mask = torch.zeros((b, max_t_audio), dtype=torch.long)
    # codebook-0 target extended by one slot so every example has room for a
    # trailing <audio_eos> right after its real last frame, however long it is.
    codebook0_ext = torch.zeros((b, max_t_audio + 1), dtype=torch.long)
    audio_mask_ext = torch.zeros((b, max_t_audio + 1), dtype=torch.long)

    for i, item in enumerate(batch):
        t = item["text_ids"].shape[0]
        text_ids[i, :t] = item["text_ids"]
        text_mask[i, :t] = 1
        a = item["codes"].shape[1]
        codes[i, :, :a] = item["codes"]
        audio_mask[i, :a] = 1
        codebook0_ext[i, :a] = item["codes"][0]
        codebook0_ext[i, a] = eos_id
        audio_mask_ext[i, : a + 1] = 1

    return {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "codes": codes,
        "audio_mask": audio_mask,
        "codebook0_ext": codebook0_ext,
        "audio_mask_ext": audio_mask_ext,
    }
