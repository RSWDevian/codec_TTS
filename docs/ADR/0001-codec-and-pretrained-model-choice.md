# ADR 0001: Codec and pretrained model choice for the initial prototype

## Status
Accepted (2026-07-17)

## Context
Needed a working codec-LM TTS prototype in one day, while keeping the workspace
structured for later custom-codec work.

## Decisions
- **Codec: Kyutai Mimi** (`kyutai/mimi` via `transformers.MimiModel`), ungated,
  8 RVQ codebooks, 2048 codes each, 12.5Hz, 24kHz audio. Matches the
  `codec/current/mimi/` folder already planned.
- **Pretrained inference: Sesame CSM-1B** (`sesame/csm-1b` via
  `transformers.CsmForConditionalGeneration`), Apache 2.0, Llama-3.2-1B
  backbone + small depth decoder producing Mimi codes. Gated on HuggingFace
  (requires accepting license terms for both `sesame/csm-1b` and
  `meta-llama/Llama-3.2-1B`).
- **Dataset: LJSpeech** for the toy training scaffold -- not in the originally
  planned `datasets/` tree (which listed librispeech/libritts/commonvoice/vctk/
  custom); added as `datasets/ljspeech/` because it's small and fast to set up.
- **Toy backbone/depth transformer**: built on small `transformers.LlamaModel`
  instances rather than a custom attention implementation, mirroring the
  Moshi/CSM design (one backbone hidden state both predicts the next frame's
  semantic token and conditions a small depth transformer for that frame's
  remaining acoustic tokens).

## Consequences
- Two independent inference backends exist (`--backend csm`, `--backend toy`)
  sharing the same `BasePipeline` interface, but not sharing weights or code
  paths beyond both ultimately decoding through Mimi.
- The unused HF `datasets` pip package was deliberately dropped from
  `requirements/train.txt` because our own top-level `datasets/` directory is
  also a Python package and would shadow it on `sys.path`.
