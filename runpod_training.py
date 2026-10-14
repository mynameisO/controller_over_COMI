"""Foreground COMI/HotpotQA workflow for an NVIDIA L40 RunPod."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent


def verify_data(directory):
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, info in manifest["files"].items():
            path = directory / name
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != info["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            print(f"{name}: {info['rows']} rows, checksum OK", flush=True)
    else:
        raise FileNotFoundError(f"Missing data manifest: {manifest_path}. Run prepare first.")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "prepare", "check", "smoke", "smoke-eval", "train", "eval"])
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--venv-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-output-dir", type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--segment-size", type=int, default=4096)
    parser.add_argument("--merge-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--min-free-disk-gib", type=float, default=50)
    parser.add_argument("--torch-version", default="2.6.0")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu124")
    return parser


def main():
    parser = make_parser()
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if extra and args.command not in {"train", "smoke", "eval", "smoke-eval"}:
        parser.error(f"Unexpected arguments: {' '.join(extra)}")
    if min(args.num_gpus, args.segment_size, args.merge_size, args.max_steps,
           args.gradient_accumulation_steps, args.logging_steps, args.save_total_limit) <= 0:
        parser.error("GPU count, lengths, steps, accumulation and checkpoint limit must be positive")
    if args.min_free_disk_gib < 0:
        parser.error("min-free-disk-gib must not be negative")
    workspace = args.workspace.resolve()
    venv = (args.venv_dir or workspace / "comi-venv-l40").resolve()
    data = (args.data_dir or workspace / "comi-data/hotpotqa").resolve()
    output = (args.output_dir or workspace / "comi-output/hotpotqa-l40").resolve()
    smoke_output = (args.smoke_output_dir or workspace / "comi-output/hotpotqa-l40-smoke").resolve()
    python = venv / "bin/python"
    env = os.environ.copy()
    env.update({
        "PATH": str(venv / "bin") + os.pathsep + env.get("PATH", ""),
        "VENV_DIR": str(venv), "TORCH_VERSION": args.torch_version,
        "TORCH_INDEX_URL": args.torch_index_url,
        "HF_HOME": env.get("HF_HOME", str(workspace / "hf-cache")),
        "HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1", "DATA_DIR": str(data),
        "TRAIN_FILE": str(data / "train.jsonl"), "TEST_FILE": str(data / "eval.jsonl"),
        "OUTPUT_DIR": str(output), "SMOKE_OUTPUT_DIR": str(smoke_output),
        "MODEL_NAME": args.model_name, "NUM_GPUS": str(args.num_gpus),
        "SEGMENT_SIZE": str(args.segment_size), "MERGE_SIZE": str(args.merge_size),
        "MAX_STEPS": str(args.max_steps),
        "GRADIENT_ACCUMULATION_STEPS": str(args.gradient_accumulation_steps),
    })

    def run(command):
        # Inherit stdout/stderr: progress and loss appear directly in the terminal.
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if args.command == "setup":
        run(["bash", "scripts/runpod_setup.sh"])
        return
    if not python.is_file():
        parser.error(f"Training environment not found: {python}. Run setup first.")
    if args.command == "prepare":
        if not data.exists():
            bundled = ROOT / "data/hotpotqa"
            if (bundled / "manifest.json").is_file():
                data.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(bundled, data)
            else:
                run([str(python), "prepare_hotpotqa.py", "--output-dir", str(data)])
        verify_data(data)
        return
    if args.command == "check":
        run([str(python), "scripts/preflight.py", "--train-file", str(data / "train.jsonl"),
             "--eval-file", str(data / "eval.jsonl"), "--num-gpus", str(args.num_gpus),
             "--output-dir", str(output)])
        return
    if args.command in {"train", "smoke"}:
        target = smoke_output if args.command == "smoke" else output
        target.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target.parent).free / 2**30
        print(f"Free output disk: {free:.1f} GiB; reserve: {args.min_free_disk_gib:g} GiB", flush=True)
        if free < args.min_free_disk_gib:
            raise RuntimeError("Insufficient free output disk; increase storage or choose another output directory.")
        command = ["bash", "scripts/runpod_train.sh"]
        if args.command == "smoke":
            command.append("smoke")
        command += ["--logging_steps", str(args.logging_steps),
                    "--save_total_limit", str(args.save_total_limit)]
        run(command + extra)
        return
    if args.command == "smoke-eval":
        env.update({"RESTORE_FROM": str(smoke_output / "checkpoint-3"),
                    "OUTPUT_DIR": str(smoke_output), "TEST_FILE": str(data / "smoke_eval.jsonl"),
                    "NUM_SAMPLES": "10"})
    else:
        env.update({"RESTORE_FROM": str(output), "TEST_FILE": str(data / "validation.jsonl"),
                    "NUM_SAMPLES": "0"})
    run(["bash", "scripts/runpod_eval.sh"] + extra)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except KeyboardInterrupt:
        raise SystemExit(130)
