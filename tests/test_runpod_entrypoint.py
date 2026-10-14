import subprocess
import sys
from pathlib import Path

import runpod_training


def capture_run(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_l40_setup_selects_cuda_stack_and_streams_output(tmp_path, monkeypatch):
    calls = capture_run(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["runpod_training.py", "setup", "--workspace", str(tmp_path)])
    runpod_training.main()
    command, kwargs = calls[0]
    assert command == ["bash", "scripts/runpod_setup.sh"]
    assert kwargs["env"]["TORCH_VERSION"] == "2.6.0"
    assert kwargs["env"]["TORCH_INDEX_URL"].endswith("/cu124")
    assert kwargs["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert "stdout" not in kwargs and "stderr" not in kwargs


def test_training_forwards_configuration_and_cli_overrides(tmp_path, monkeypatch):
    python = tmp_path / "comi-venv-l40/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls = capture_run(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "runpod_training.py", "train", "--workspace", str(tmp_path),
        "--segment-size", "1024", "--min-free-disk-gib", "0",
        "--", "--learning_rate", "2e-5",
    ])
    runpod_training.main()
    command, kwargs = calls[0]
    assert command == ["bash", "scripts/runpod_train.sh", "--logging_steps", "10",
                       "--save_total_limit", "1", "--learning_rate", "2e-5"]
    assert kwargs["env"]["SEGMENT_SIZE"] == "1024"
    assert kwargs["env"]["MAX_STEPS"] == "10000"
    assert kwargs["env"]["GRADIENT_ACCUMULATION_STEPS"] == "8"
    assert "stdout" not in kwargs and "stderr" not in kwargs


def test_smoke_eval_uses_intermediate_checkpoint(tmp_path, monkeypatch):
    python = tmp_path / "comi-venv-l40/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls = capture_run(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["runpod_training.py", "smoke-eval", "--workspace", str(tmp_path)])
    runpod_training.main()
    command, kwargs = calls[0]
    assert command == ["bash", "scripts/runpod_eval.sh"]
    assert Path(kwargs["env"]["RESTORE_FROM"]).parts[-2:] == ("hotpotqa-l40-smoke", "checkpoint-3")
    assert kwargs["env"]["NUM_SAMPLES"] == "10"
