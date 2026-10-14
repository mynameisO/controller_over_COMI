#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
export NUM_GPUS="${NUM_GPUS:-1}"
export TEST_FILE="${TEST_FILE:-/workspace/comi-data/hotpotqa/validation.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/comi-output/hotpotqa}"
export RESTORE_FROM="${RESTORE_FROM:-${OUTPUT_DIR}}"
export NUM_SAMPLES="${NUM_SAMPLES:-0}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-4096}"
bash test.sh --max_new_tokens 128 "$@"
