# speech-foundation

Codec-LM based TTS. Two inference backends today:

- `--backend csm` — pretrained Sesame CSM-1B (real, intelligible speech; needs HF auth, see below).
- `--backend toy` — our own from-scratch backbone + depth transformer, trained on a tiny LJSpeech subset (proves the training pipeline; audio quality is expected to be poor/babble).

See `docs/architecture/system.md` for the architecture, `docs/roadmap.md` for what's implemented vs. stubbed, and `docs/ADR/0001-codec-and-pretrained-model-choice.md` for why.

## Setup

```
bash scripts/setup.sh --with-ljspeech
```

Then, only if you want `--backend csm`: accept the license terms at
[huggingface.co/sesame/csm-1b](https://huggingface.co/sesame/csm-1b) and
[huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B),
then `.venv/bin/hf auth login`.

## Toy training + inference (no HF auth needed)

```
.venv/bin/python scripts/tokenize_dataset.py --config configs/datasets/ljspeech.yaml
.venv/bin/python scripts/train_backbone.py --config configs/training/toy_overfit.yaml
.venv/bin/python scripts/inference.py --backend toy --text "hello there" --output outputs/audio/toy_test.wav
```

## Pretrained inference (needs HF auth)

```
.venv/bin/python scripts/inference.py --backend csm --text "Hello from Sesame." --output outputs/audio/csm_test.wav
```

## Tests

```
.venv/bin/pytest tests/ -m "not network"   # fast, no downloads
.venv/bin/pytest tests/                     # includes Mimi encode/decode roundtrip
```
