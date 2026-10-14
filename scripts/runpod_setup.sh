#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV_DIR="${VENV_DIR:-/workspace/comi-venv}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --upgrade --no-cache-dir "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements-runpod.txt
python -m pip check
python -m pip freeze > "$VENV_DIR/environment.txt"
export HF_HUB_ENABLE_HF_TRANSFER=0
python scripts/check_cuda.py
echo "Setup complete. Run: source $VENV_DIR/bin/activate"
