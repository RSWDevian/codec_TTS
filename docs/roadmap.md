# Roadmap

## Today's prototype (done)
- Pretrained inference demo (`--backend csm`, Sesame CSM-1B).
- Toy from-scratch training scaffold (`--backend toy`): Mimi tokenization of a
  32-clip LJSpeech subset, joint backbone+depth transformer training, decode
  back to audio. Proves the pipeline, not audio quality.

## Known open items / tree ambiguities (not resolved, flagging for later)
- `codec/evaluation/` vs. top-level `evaluation/codec/` overlap in purpose --
  pick one home for codec reconstruction-quality metrics.
- Top-level `experiments/` vs. `docs/experiments/` overlap -- pick one home
  for experiment tracking/notes.
- `scripts/benchmark.py` was not created today (no benchmarking deliverable
  was in scope) -- add it alongside `evaluation/benchmark/` when needed.

## Future work (see per-directory READMEs for detail)
- Custom codec (`codec/future/`), once the pretrained-Mimi prototype is
  validated.
- Alternative backbones: SSM / hybrid (`models/backbone/ssm|hybrid/`).
- Streaming inference, since Mimi's encoder/decoder are already
  causal/streaming-capable (`inference/streaming/`, `kv_cache/`, `chunking/`).
- Additional dataset loaders (LibriSpeech, LibriTTS, Common Voice, VCTK,
  custom) following `datasets/loaders/base.py`'s `TTSDatasetLoader` interface.
- Serving (`serving/`), evaluation harnesses (`evaluation/`), decoupled depth
  fine-tuning (`scripts/train_depth.py`).
