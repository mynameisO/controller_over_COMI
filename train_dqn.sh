#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-./data/train.jsonl}"
RESTORE_FROM="${RESTORE_FROM:-/path/to/comi_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dqn}"
DQN_CHECKPOINT="${DQN_CHECKPOINT:-${OUTPUT_DIR}/dqn_controller.pt}"
MERGE_SIZES="${MERGE_SIZES:-16,32}"
SEGMENT_SIZE="${SEGMENT_SIZE:-10000}"
DQN_EPOCHS="${DQN_EPOCHS:-5}"
DQN_BATCH_SIZE="${DQN_BATCH_SIZE:-1}"
DQN_NUM_TRAIN="${DQN_NUM_TRAIN:-0}"
DQN_EPSILON_DECAY_STEPS="${DQN_EPSILON_DECAY_STEPS:-0}"

# One process: COMI stays frozen; only the DQN controller is trained.
# An existing DQN_CHECKPOINT is loaded to resume training.
exec python dqn_over_comi_train.py \
  --model_name_or_path "${MODEL_NAME}" \
  --restore_from "${RESTORE_FROM}" \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --dqn_checkpoint "${DQN_CHECKPOINT}" \
  --merge_sizes "${MERGE_SIZES}" \
  --segment_size "${SEGMENT_SIZE}" \
  --dqn_epochs "${DQN_EPOCHS}" \
  --dqn_batch_size "${DQN_BATCH_SIZE}" \
  --dqn_num_train "${DQN_NUM_TRAIN}" \
  --dqn_epsilon_decay_steps "${DQN_EPSILON_DECAY_STEPS}" \
  --report_to none "$@"
