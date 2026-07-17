from __future__ import annotations

import yaml


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
