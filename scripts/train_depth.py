#!/usr/bin/env python
"""Not exercised today -- joint training in scripts/train_backbone.py is
sufficient to prove the pipeline. Wired for future decoupled depth
fine-tuning: load a backbone checkpoint, optionally freeze it, and continue
training only the depth transformer via the same training/trainer.py loop.
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a backbone+depth checkpoint")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.parse_args()
    raise NotImplementedError(
        "Decoupled depth training is not implemented today -- see scripts/train_backbone.py "
        "for the joint training path exercised in the current prototype."
    )


if __name__ == "__main__":
    main()
