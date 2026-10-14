# Train COMI on HotpotQA using RunPod

For L40 and the foreground script workflow, follow [L40.md](L40.md) and run
`python3 runpod_training.py --help`. Training logs are shown directly in the terminal.

Backbone: `Qwen/Qwen3-4B-Instruct-2507`. Encoder: first 8 layers,
memory fusion: 1 layer, decoder: full backbone with LoRA (rank 128).
Encoder and fusion are fully trained. Compression is about 16x by default.
COMI produces embedding memories consumed by its own decoder, not compressed text.

For the older Blackwell notebook workflow, upload `runpod_training_B300.ipynb` and `comi-runpod.zip`
to `/workspace`, open the notebook in JupyterLab, and execute cells in order.
It installs an isolated training venv, verifies data, runs smoke tests, starts
full training in the background, and provides monitoring/evaluation cells.
The notebook kernel can remain Python 3; training uses the venv via subprocess.

## 1. Pod and storage

Use a Linux CUDA Pod with Python 3.10 through 3.12 and a driver compatible with
CUDA 12.9. Start with one BF16-capable 80 GB GPU (A100/H100) and at least
150 GB of storage under `/workspace`. This is a planning recommendation,
not a measured memory requirement; run the smoke test on your chosen GPU.
The default workflow uses PyTorch SDPA, so FlashAttention compilation is unnecessary.

Attach a network volume if checkpoints must survive Pod deletion. A normal
volume under `/workspace` survives stop/restart but is deleted with its Pod.
See [RunPod storage](https://docs.runpod.io/pods/storage/types).

Upload `artifacts/comi-runpod.zip` and extract it under `/workspace/comi`, or
copy this branch's complete working tree. No Git push is required for the ZIP path.

```bash
mkdir -p /workspace/comi
unzip /workspace/comi-runpod.zip -d /workspace/comi
cd /workspace/comi
bash scripts/runpod_setup.sh
source /workspace/comi-venv/bin/activate
export HF_HOME=/workspace/hf-cache
```

The setup installs PyTorch 2.8.0/cu129 and the pinned requirements in an isolated
venv. Do not reuse the original README's older mixed dependency stack.
The exact installed dependency versions are written to `environment.txt` inside the training venv.
Local validation details are in [VALIDATION.md](VALIDATION.md).

## 2. Data

The ZIP includes converted `data/hotpotqa` when packaged after preparation.
Copy it to the persistent data location:

```bash
mkdir -p /workspace/comi-data
cp -a data/hotpotqa /workspace/comi-data/
```

Alternatively download and convert directly on the Pod:

```bash
python prepare_hotpotqa.py --output-dir /workspace/comi-data/hotpotqa
```

The converter resolves the dataset revision to a commit and records SHA256
checksums, counts, and provenance in `manifest.json`. It refuses to overwrite
existing data unless `--force` is supplied. All titles and paragraphs are retained;
gold supporting facts are not used to filter context.

Source: [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
CC BY-SA 4.0. `distractor` has 90,447 training and 7,405 validation examples.
Files: `train.jsonl`, `validation.jsonl`, `eval.jsonl` (512 seeded validation
examples), `smoke_train.jsonl` (32), and `smoke_eval.jsonl` (10).
The eval and smoke files overlap validation; they are not independent test sets.
Reserve an independent holdout if you tune extensively against validation.

## 3. Tests and GPU smoke run

```bash
python -m pytest -q
bash scripts/runpod_train.sh smoke
RESTORE_FROM=/workspace/comi-output/hotpotqa-smoke \
  OUTPUT_DIR=/workspace/comi-output/hotpotqa-smoke \
  NUM_SAMPLES=10 bash scripts/runpod_eval.sh
```

The CPU tests use a tiny randomly initialized Qwen3 model; the GPU smoke run
uses the actual 4B backbone and verifies training, evaluation, and saving.
The first real-model run downloads the model weights. Check loss is finite,
`checkpoint-3` exists, and inference finishes before starting the full job.

## 4. Full training

```bash
bash scripts/runpod_train.sh
```

Defaults: one GPU, BF16, batch 1, gradient accumulation 8, learning rate 1e-5,
10,000 optimizer steps, evaluation every 1,000 and saving every 2,000 steps,
two retained intermediate checkpoints, gradient checkpointing, merge size 16,
context segments of 4,096 tokens. Periodic evaluation reports answer loss;
post-training inference reports answer EM/F1. W&B is disabled.

Change runtime options with environment variables or additional arguments:

```bash
# Two GPUs: effective batch becomes 1 * 2 * 8 = 16.
NUM_GPUS=2 bash scripts/runpod_train.sh

# Lower optimizer/gradient memory with LoRA on encoder and fusion too.
OUTPUT_DIR=/workspace/comi-output/hotpotqa-lora \
  bash scripts/runpod_train.sh \
  --full_finetune_encoder false --lora_encoder true \
  --full_finetune_lsa false --lora_lsa true

# Train multiple compression ratios; architecture is saved with the checkpoint.
OUTPUT_DIR=/workspace/comi-output/hotpotqa-random \
  bash scripts/runpod_train.sh --is_random true --merge_sizes 8,16,32
```

DDP replicates the model on every GPU; more GPUs do not pool their VRAM.
If out of memory, lower `SEGMENT_SIZE`, then consider LoRA on all parts.
Encoder/fusion call their transformer backbones directly to avoid unused LM logits,
but long inputs still consume substantial memory for activations and attention.
`model_max_length` is not enforced by tokenization; segment size does not truncate
the total context. Inspect memory with `nvidia-smi` during the real-model smoke run.

Optional DeepSpeed ZeRO-2 CPU optimizer offload (not part of local CPU validation):

```bash
python -m pip install deepspeed==0.18.7
NUM_GPUS=2 bash scripts/runpod_train.sh --deepspeed ds_config.json
```

For long jobs use your Pod's terminal multiplexer, for example `tmux new -s comi`,
then run the command inside it. Logs are displayed directly in the terminal:

```bash
bash scripts/runpod_train.sh
```

## 5. Resume and evaluation

Re-running the same training command auto-resumes the latest `checkpoint-N`,
including optimizer/scheduler/RNG state. Output directories without resumable
checkpoints are refused when non-empty. To start over, use a new `OUTPUT_DIR`.
Explicit resume:

```bash
bash scripts/runpod_train.sh \
  --resume_from_checkpoint /workspace/comi-output/hotpotqa/checkpoint-8000
```

`--restore_from` only loads model weights for warm-starting; it does not restore
optimizer state. Keep the same backbone/architecture when resuming.

```bash
bash scripts/runpod_eval.sh
MERGE_SIZE=32 OUTPUT_DIR=/workspace/comi-output/hotpotqa-eval32 \
  RESTORE_FROM=/workspace/comi-output/hotpotqa bash scripts/runpod_eval.sh
```

Inference restores architecture/LoRA settings from `comi_model_args.json` and
checks all weight keys strictly. Use the same `MODEL_NAME` as training.
Inference is deterministic and generates up to 128 new tokens. Full validation
is used by default. Output files are under `eval_result/`; their legacy `nq_`
prefix is retained, even for HotpotQA. Scores cover answers only, not HotpotQA's
supporting-fact or joint metrics.

Back up the final output and checkpoints before deleting the Pod. Final output
contains `model.safetensors`, `comi_model_args.json`, tokenizer files, training
arguments, and metrics. Intermediate checkpoints additionally include optimizer,
scheduler, trainer state, and RNG state for exact continuation.
