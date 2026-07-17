# models

Current (used by today's toy prototype):
- `backbone/transformer/` — toy backbone (small `LlamaModel`), predicts codebook-0 tokens per frame conditioned on text.
- `depth_transformer/` — toy depth transformer, predicts codebooks 1-7 per frame conditioned on backbone hidden state.
- `embeddings/`, `heads/`, `common/` — shared codec-embedding tables, LM heads, and utility helpers reused by both models above.

Future:
- `backbone/ssm/`, `backbone/hybrid/` — alternative backbone architectures (state-space / hybrid SSM+attention) to explore once the transformer backbone baseline is validated.
- `voice/`, `emotion/` — conditioning modules for voice cloning / emotion control.
- `predictor/` — reserved for auxiliary prediction heads (e.g. duration/prosody).
