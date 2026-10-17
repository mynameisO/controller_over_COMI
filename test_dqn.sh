#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TEST_FILE="${TEST_FILE:-./data/test.jsonl}"
RESTORE_FROM="${RESTORE_FROM:-/path/to/comi_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dqn}"
DQN_CHECKPOINT="${DQN_CHECKPOINT:-${OUTPUT_DIR}/dqn_controller.pt}"
MERGE_SIZES="${MERGE_SIZES:-16,32}"
SEGMENT_SIZE="${SEGMENT_SIZE:-10000}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

# Evaluate the greedy DQN policy and write predictions plus EM/F1 metrics.
exec python dqn_over_comi_infer.py \
  --dqn_mode eval \
  --model_name_or_path "${MODEL_NAME}" \
  --restore_from "${RESTORE_FROM}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --dqn_checkpoint "${DQN_CHECKPOINT}" \
  --merge_sizes "${MERGE_SIZES}" \
  --segment_size "${SEGMENT_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --num_samples "${NUM_SAMPLES}" \
  --report_to none "$@"
