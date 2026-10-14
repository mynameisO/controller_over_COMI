# Preparation validation

Validated on 2026-10-13 on Windows/Python 3.10 in an isolated CPU environment.
No production 4B model training was performed locally.

- PyTorch 2.6.0+cpu; Transformers 4.57.5; PEFT 0.17.1; Accelerate 1.11.0;
  Datasets 4.3.0; Safetensors 0.7.0. `pip check` passed.
- Four pytest cases passed: HotpotQA conversion including distractors and invalid
  answer rejection; masks preserving real EOS tokens; tiny Qwen3 training,
  checkpoint reload, generation, and optimizer resume with default tuning;
  the same lifecycle with LoRA on encoder, fusion, and decoder.
- Both training modes updated encoder weights/adapters. Reloaded tensors matched
  saved model tensors exactly. Resumed trainer state reached the next global step.
- `instruction_finetune.py --help` and `ft_inference.py --help` parsed successfully.
- Bash syntax checked for all five launch/setup scripts. Extra training CLI
  arguments were verified through a mocked `torchrun` invocation.
- All converted JSONL rows passed schema/non-empty validation: train 90,447;
  validation 7,405; periodic eval 512; smoke train 32; smoke eval 10.
- Source dataset commit: `1908d6afbbead072334abe2965f91bd2709910ab`.
  Per-file counts and SHA256 hashes are recorded in `data/hotpotqa/manifest.json`.

Pending on the chosen RunPod GPU: CUDA/BF16 preflight, real Qwen3-4B smoke run,
actual VRAM measurement, and full training/evaluation. Multi-GPU NCCL and optional
DeepSpeed have not been exercised on the local CPU environment.

Reproduce tests with `python -m pytest -q`. Build the transport archive with
`python scripts/package_runpod.py`; the builder verifies data hashes and ZIP CRCs.

## Blackwell environment correction

The RunPod setup now installs PyTorch 2.8.0/cu129, disables the template's optional
`HF_HUB_ENABLE_HF_TRANSFER` path, and checks actual BF16 forward/backward,
embedding, and SDPA kernels before loading weights. Python 3.12 is supported by
this wheel stack. The four existing CPU regression cases still pass on the local
PyTorch 2.6.0 CPU environment; CUDA 12.9 and the user's Blackwell device require
validation on the Pod. The original setup's CUDA 12.4 wheel was incompatible with
the sm_120 GPU shown in the user's log.
