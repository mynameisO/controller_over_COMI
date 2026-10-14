#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv /workspace/comi-venv
source /workspace/comi-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-runpod.txt
python -m pip check
python -m pip freeze > /workspace/comi-environment.txt
export HF_HUB_ENABLE_HF_TRANSFER=0
python scripts/check_cuda.py
echo 'Setup complete. Run: source /workspace/comi-venv/bin/activate'
