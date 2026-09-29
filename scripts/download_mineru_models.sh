#!/usr/bin/env bash
# Download MinerU 1.x's model weights and write its config, on the pod's volume.
#
#   bash scripts/download_mineru_models.sh          (pod_bootstrap.sh runs it)
#
# MinerU's own download_models_hf.py is used, from the release matching the
# installed magic-pdf. It fetches a config template from MinerU's master branch,
# which MinerU 2.x no longer has (404): that URL is pointed at the same release.
# The weights go to HF_HOME (on the volume); the config the script writes to
# ~/magic-pdf.json is moved to MINERU_TOOLS_CONFIG_JSON (on the volume), and its
# device-mode set to cuda - MinerU's default is cpu.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKSPACE="${FIDEON_WORKSPACE:-/workspace}"
VENV_OCR="$WORKSPACE/venv-ocr"
[ -x "$VENV_OCR/bin/python" ] || { echo "download_mineru_models: no OCR environment at $VENV_OCR" >&2; exit 1; }
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
# RunPod images turn on the hf_transfer downloader for every process; it is not
# installed in these environments, and Hugging Face then refuses to download.
export HF_HUB_ENABLE_HF_TRANSFER=0
export MINERU_TOOLS_CONFIG_JSON="${MINERU_TOOLS_CONFIG_JSON:-$WORKSPACE/magic-pdf.json}"

version="$("$VENV_OCR/bin/python" -c 'import importlib.metadata as m; print(m.version("magic-pdf"))')"
tag="magic_pdf-${version}-released"
echo "MinerU (magic-pdf) $version: model weights into $HF_HOME"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "https://raw.githubusercontent.com/opendatalab/MinerU/$tag/scripts/download_models_hf.py" -o "$tmp/download_models_hf.py"
sed -i "s#MinerU/raw/master/magic-pdf.template.json#MinerU/raw/$tag/magic-pdf.template.json#" "$tmp/download_models_hf.py"
grep -q "raw/$tag/magic-pdf.template.json" "$tmp/download_models_hf.py" \
  || { echo "download_mineru_models: the template URL in MinerU's script changed; update this script" >&2; exit 1; }
(cd "$tmp" && "$VENV_OCR/bin/python" download_models_hf.py)

# The script writes ~/magic-pdf.json (the container disk): keep it on the volume.
if [ "$HOME/magic-pdf.json" != "$MINERU_TOOLS_CONFIG_JSON" ] && [ -f "$HOME/magic-pdf.json" ]; then
  mv -f "$HOME/magic-pdf.json" "$MINERU_TOOLS_CONFIG_JSON"
fi
"$VENV_OCR/bin/python" -m data_pipeline.ocr.mineru_config --cuda
echo "download_mineru_models: done ($MINERU_TOOLS_CONFIG_JSON)"
