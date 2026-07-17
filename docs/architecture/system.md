# System Architecture

Codec-LM TTS: text -> [backbone transformer predicts codebook-0 tokens per frame] -> [depth transformer predicts codebooks 1-7 per frame, conditioned on backbone hidden state] -> Mimi decoder -> audio.

Today's prototype runs two independent paths sharing this shape:
- Pretrained: `sesame/csm-1b` (Llama-3.2-1B backbone + small depth decoder + bundled Mimi), via `inference/pipeline/csm_pipeline.py`.
- Toy/from-scratch: small `LlamaModel`-based backbone + depth transformer, trained on a LJSpeech subset, via `training/trainer.py` and `inference/pipeline/toy_pipeline.py`.

See `docs/roadmap.md` for what's stubbed vs. implemented, and `docs/ADR/0001-codec-and-pretrained-model-choice.md` for why these choices were made.
