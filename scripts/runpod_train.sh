#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export DATA_DIR="${DATA_DIR:-/workspace/comi-data/hotpotqa}"
export NUM_GPUS="${NUM_GPUS:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-4096}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/comi-output/hotpotqa}"
export TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
export TEST_FILE="${TEST_FILE:-${DATA_DIR}/eval.jsonl}"
if [[ "${1:-}" == "smoke" ]]; then
  shift
  export TRAIN_FILE="${DATA_DIR}/smoke_train.jsonl"
  export TEST_FILE="${DATA_DIR}/smoke_eval.jsonl"
  export OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/workspace/comi-output/hotpotqa-smoke}"
  export MAX_STEPS=3 SAVE_STEPS=3 EVAL_STEPS=3
  export GRADIENT_ACCUMULATION_STEPS=1
fi
python scripts/preflight.py --train-file "$TRAIN_FILE" --eval-file "$TEST_FILE" --num-gpus "$NUM_GPUS" --output-dir "$OUTPUT_DIR"
bash train.sh "$@"
