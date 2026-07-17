# codec

- `current/` — the codec actually in use today: a thin wrapper around Kyutai's pretrained Mimi (`transformers.MimiModel`). See `current/wrapper.py`.
- `future/` — home for a custom-trained codec, once we outgrow Mimi.
- `tokenizer/` — reserved for a unified encode/decode CLI-facing tokenizer abstraction if we support multiple codecs later.
- `evaluation/` — reserved for codec-specific metrics (reconstruction quality, bitrate/quality tradeoffs). Note: overlaps in purpose with top-level `evaluation/codec/`; not yet reconciled, see `docs/roadmap.md`.
