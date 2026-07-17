#!/usr/bin/env bash
# Bootstraps the dev environment: venv, torch (CUDA), and project requirements.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "Creating virtualenv at .venv (bootstraps its own pip)..."
  python3 -m venv .venv
fi

PIP=".venv/bin/pip"
PY=".venv/bin/python"

"$PIP" install --upgrade pip

echo "Installing torch + torchaudio (matched CUDA 12.4 wheels)..."
"$PIP" install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing project requirements..."
"$PIP" install -r requirements/dev.txt

cat <<'EOF'

============================================================
NEXT STEP (manual, blocks pretrained CSM inference only):

  1. Create/use a HuggingFace account.
  2. Accept the license terms at:
       https://huggingface.co/sesame/csm-1b
       https://huggingface.co/meta-llama/Llama-3.2-1B
  3. Authenticate:
       .venv/bin/hf auth login
     (or export HF_TOKEN=... before running scripts/inference.py --backend csm)

The toy training/inference path (--backend toy) does not need this --
it only uses the ungated kyutai/mimi codec.
============================================================
EOF

if [ "${1:-}" = "--with-ljspeech" ]; then
  echo "Downloading LJSpeech-1.1 (~2.6GB)..."
  mkdir -p data/raw
  curl -L -o data/cache/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  tar -xjf data/cache/LJSpeech-1.1.tar.bz2 -C data/raw
  mv data/raw/LJSpeech-1.1 data/raw/ljspeech
fi
