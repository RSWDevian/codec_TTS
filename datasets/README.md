# datasets

Current: `loaders/base.py` defines the `TTSDatasetLoader` interface; `ljspeech/` is the one loader implemented today (see its own README — it's an addition to the original planned tree, added because the training scaffold targets LJSpeech).

Future: `librispeech/`, `libritts/`, `commonvoice/`, `vctk/`, `custom/` — additional dataset loaders following the same `TTSDatasetLoader` interface, not implemented yet.
