"""Not implemented today. Mimi's decoder supports causal/streaming generation
(frame-by-frame decode without needing the full sequence), which will matter
for inference/streaming/ and inference/kv_cache/ later. Today's prototype only
uses offline (whole-utterance) decode via codec/current/decode.py.
"""
