#!/usr/bin/env bash
# Install this repo's dependencies on a RunPod pod, in the order that works.
#
#   bash scripts/setup_pod.sh ocr|train|serve|quantize|dev
#
# Run from the repo root. Versions are declared in pyproject.toml; the
# requirements-*.txt files choose the groups per pod.
set -euo pipefail

role="${1:-}"
case "$role" in
  ocr)   requirements="requirements-ocr.txt" ;;
  train) requirements="requirements-train.txt" ;;
  serve) requirements="requirements-serve.txt" ;;
  quantize) requirements="requirements-quantize.txt" ;;
  dev)   requirements="requirements.txt" ;;
  *) echo "usage: bash scripts/setup_pod.sh ocr|train|serve|quantize|dev" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install -r "$requirements"

if [ "$role" = "train" ]; then
  # Built against the torch just installed. With build isolation pip compiles it
  # against a throwaway torch in a temporary env, which fails or produces a
  # library that does not load. MAX_JOBS bounds the compile's memory use.
  MAX_JOBS="${MAX_JOBS:-8}" python -m pip install "flash-attn>=2.7" --no-build-isolation
fi

python - "$role" <<'PY'
import importlib.metadata as md
import sys

role = sys.argv[1]
wanted = {
    "ocr": ["magic-pdf", "torch", "azure-storage-blob"],
    "train": ["torch", "transformers", "ms-swift", "peft", "vllm", "flash-attn",
              "deepspeed", "qwen-vl-utils", "azure-storage-blob"],
    "serve": ["vllm", "transformers", "azure-storage-blob"],
    "quantize": ["llmcompressor", "torch", "azure-storage-blob"],
    "dev": ["pytest", "ruff", "azure-storage-blob"],
}[role]
missing = []
for name in wanted:
    try:
        print(f"  {name:22} {md.version(name)}")
    except md.PackageNotFoundError:
        missing.append(name)
if missing:
    sys.exit(f"missing after install: {missing}")
if role in ("ocr", "train", "serve", "quantize"):
    import torch

    if not torch.cuda.is_available():
        sys.exit("torch is installed but sees no CUDA device - wrong torch build for this pod?")
    print(f"  CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
PY

if [ "$role" = "ocr" ]; then
  echo "Next: download MinerU's model weights (see the MinerU 1.x docs: download_models_hf.py),"
  echo "which writes ~/magic-pdf.json. OCR fails at model load without them."
fi
if [ "$role" = "train" ] || [ "$role" = "serve" ]; then
  echo "Next: put the base model under /workspace/models (or set FIDEON_BASE_MODEL_DIR),"
  echo "pin its revision in configs/base_model.yaml, and run: python scripts/phase0_spike.py"
fi
echo "setup_pod: $role ready"
