# inference

Current: `pipeline/` holds the two inference backends used today (`csm_pipeline.py` for pretrained CSM-1B, `toy_pipeline.py` for our own toy backbone+depth model), both driven by `scripts/inference.py`.

Future: `streaming/`, `kv_cache/`, `chunking/` — low-latency streaming generation (Mimi's encoder/decoder are already causal/streaming-capable, not yet exploited here). `websocket/`, `grpc/` — network-facing inference servers. `runtime/` — reserved for runtime-level optimizations (quantization, batching).
