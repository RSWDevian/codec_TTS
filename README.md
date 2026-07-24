# Codec_TTS

Codec-LM based Text-to-Speech for Hindi, using a dual-transformer architecture (Backbone + Depth Transformer) with the Kyutai Mimi neural codec (8 codebooks, 12.5 Hz frame rate, 24 kHz audio).

- [Architecture](#architecture)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Inference Parameters](#inference-parameters)
- [Repository Structure](#repository-structure)

## Architecture

### Overview

The model is a **two-stage autoregressive audio language model** inspired by Moshi/CSM:

1. **Backbone Transformer** — predicts **codebook 0** (the primary audio stream) autoregressively, conditioned on text tokens.
2. **Depth Transformer** — for each frame, predicts **codebooks 1 through 7** given the backbone's hidden state, using a per-frame autoregressive loop.

### Backbone Transformer

An autoregressive causal LLaMA model that consumes text tokens followed by audio codebook-0 tokens.

| Component | Description |
|---|---|
| Text embedding | `nn.Embedding(text_vocab_size, hidden_size)` |
| Audio embedding | Shared `MimiCodebookEmbeddings` — codebook 0 embedding table |
| Audio BOS | Learned begin-of-audio embedding prepended before audio frames |
| Core | HuggingFace `LlamaModel` (causal) |
| LM Head | `nn.Linear(hidden_size, 2049)` — 2048 Mimi codes + 1 `<audio_eos>` token |

**Forward:** `text_ids` + `codebook0_ids` → embed → concat → LLaMA → hidden states → LM head → codebook-0 logits.

**Config (`configs/backbone/hindi_backbone.yaml`):**

| Parameter | Value |
|---|---|
| `hidden_size` | 768 |
| `num_hidden_layers` | 13 |
| `num_attention_heads` | 12 |
| `intermediate_size` | 2688 |
| `max_position_embeddings` | 2048 |
| `audio_vocab_size` | 2049 |

### Depth Transformer

A small per-frame autoregressive LLaMA model. Frames are independent — during training it receives 6 teacher-forced codebook tokens (1..6) and predicts all 7 codebooks (1..7) at once. During inference it runs autoregressively for 7 steps per frame.

**Config (`configs/backbone/hindi_depth_config.yaml`):**

| Parameter | Value |
|---|---|
| `hidden_size` | 384 |
| `num_hidden_layers` | 6 |
| `num_attention_heads` | 6 |
| `intermediate_size` | 1536 |
| `num_depth_codebooks` | 7 |

### Mimi Codec

The audio tokenizer/detokenizer is [Kyutai Mimi](https://huggingface.co/kyutai/mimi) — a 8-codebook residual vector quantizer running at 12.5 Hz on 24 kHz audio.

### Parameter Count

- Backbone: ~47M
- Depth Transformer: ~10M
- **Total: ~57M parameters**

## Training the Model

### Dependencies

```bash
source .venv/bin/activate
pip install -r requirements/train.txt
```

### Training Config (`configs/training/hindi_training.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `seed` | 42 | |
| `batch_size` | 1 | Per GPU |
| `gradient_accumulation_steps` | 32 | Effective batch = 32 |
| `lr` | 3e-4 | AdamW, weight decay 0.01 |
| `warmup_steps` | 1000 | Linear warmup + cosine schedule |
| `num_steps` | 100000 | Total optimizer steps |
| `checkpoint_every` | 5000 | Saves `checkpoints/hindi/step_{N}.pt` |
| `val_every` | 2000 | Validation loss on 3% held-out set |
| `max_audio_frames` | 1500 | Max frames per utterance |
| `num_codebooks` | 8 | |

Training uses `bfloat16` AMP, gradient checkpointing, and gradient clipping at norm 1.0.

### Steps

#### 1. Prepare Data

**Option A — Record your own voice:**

```bash
python scripts/record_studio.py
# open http://localhost:7861
```

Uses a local Ollama model to generate Hindi sentences, then records audio via microphone. Saves `data/raw/hindi_iisc/NNN.wav` + `NNN.txt`.

**Option B — Use the IISc Hindi dataset:**

```bash
python scripts/prepare_iisc_dataset.py
```

Extracts `somu9/iisc_mono_hindi_female` from HuggingFace cache.

#### 2. Tokenize Dataset

```bash
python scripts/tokenize_dataset.py --config configs/datasets/hindi.yaml
```

Encodes audio through `MimiCodec` → codec codes, tokenizes text via `HindiTokenizer` → text IDs. Saves to `data/tokenized/hindi_iisc/{utt_id}.pt`.

#### 3. Train

```bash
python scripts/train.py --config configs/training/hindi_training.yaml
```

Trains both backbone and depth transformer jointly. Checkpoints are saved to `checkpoints/hindi/`.

#### 4. Monitor

```bash
python scripts/monitor.py
# open http://localhost:7860
```

Gradio dashboard showing loss curves (total, backbone, depth) from TensorBoard logs.

## Testing the Model

### Quick Inference

```bash
python scripts/inference.py --backend hindi --text "नमस्ते" --output outputs/audio/hindi_test.wav
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--backend` | `hindi` | `hindi` or `csm` (pretrained Sesame CSM-1B) |
| `--text` | required | Input text to synthesize |
| `--output` | `outputs/audio/output.wav` | Output WAV path |
| `--tokenizer-type` | `hindi` | Tokenizer backend (`hindi` or `grapheme`) |
| `--hindi-config` | `configs/inference/hindi_default.yaml` | Inference parameters |
| `--dataset-config` | `configs/datasets/hindi.yaml` | Dataset paths (vocab, manifest) |
| `--backbone-config` | `configs/backbone/hindi_backbone.yaml` | Backbone architecture config |
| `--depth-config` | `configs/backbone/hindi_depth_config.yaml` | Depth architecture config |

### Inference Flow

1. Text is tokenized via `HindiTokenizer` → `[BOS, char_ids..., EOS]`
2. Frame-by-frame autoregressive generation:
   - **Backbone** generates next codebook-0 token given text + previous audio codes
   - If token is `<audio_eos>` (id 2048), generation stops
   - **Depth Transformer** generates codebooks 1..7 for the frame
3. All 8 codebook streams are decoded through `MimiCodec.decode()` → waveform

## Inference Parameters

Defined in `configs/inference/hindi_default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `checkpoint_dir` | `checkpoints/hindi` | Directory containing `step_*.pt` checkpoints |
| `device` | `cuda` | Inference device |
| `frames_per_char` | 1.65 | Safety cap: `max_frames = min(400, max(8, len(text) * 1.65))` |
| `max_frames` | 400 | Absolute upper bound on generated frames |
| `top_k` | 10 | Top-k sampling (number of candidates) |
| `temperature` | 0.8 | Sampling temperature |
| `repetition_penalty` | 1.3 | Penalty applied to recently seen tokens (last 20) |

## Repository Structure

```
codec_TTS/
  codec/current/            MimiCodec wrapper
  configs/                   YAML configs (backbone, depth, dataset, inference, training)
  data/                      Raw audio, tokenized data, metadata (gitignored)
  datasets/                  Dataset loaders
  inference/pipeline/        Inference pipelines (hindi, csm)
  models/
    backbone/transformer/    HindiBackbone (13-layer LLaMA)
    depth_transformer/       HindiDepthTransformer (6-layer LLaMA)
    embeddings/              Shared MimiCodebookEmbeddings
    heads/                   LM head projection
  preprocessing/text/        Tokenizers (hindi, grapheme) + sentence generator
  requirements/              Dependency files
  scripts/                   CLI entry points (train, inference, record, monitor, etc.)
  training/                  Training loop, dataset, callbacks, losses, optimizer, scheduler
  tests/                     Unit tests
  backup_models/             Saved trained model checkpoints + configs
  checkpoints/               Training checkpoints (gitignored)
  outputs/                   Logs, audio samples, figures (gitignored)
```
