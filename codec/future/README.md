# codec/future

Placeholder for a custom-trained neural audio codec, once the prototype is validated on the pretrained Mimi codec (see `codec/current/`).

- `encoder/`, `decoder/` — SEANet-style convolutional encoder/decoder
- `rvq/` — residual vector quantizer
- `seanet/` — shared SEANet building blocks used by encoder/decoder
- `transformer/` — optional transformer bottleneck (as in Mimi/EnCodec)
- `discriminator/` — adversarial discriminators for perceptual training
- `trainer/` — training loop for the codec itself (distinct from `training/codec/`, which will hold configs/entrypoints that call into this)
