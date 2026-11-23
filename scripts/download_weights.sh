#!/usr/bin/env bash
# Download NavGPT3 VLA weights into models/ (run on the GPU server).
# Weights are AGPL-3.0; see the Hugging Face model cards.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v hf >/dev/null || pip install -U huggingface_hub
SIZES="${*:-8B 4B}"
for size in $SIZES; do
  hf download "Metacognition-AI/NavGPT3-${size}" --local-dir "models/NavGPT3-${size}"
done
