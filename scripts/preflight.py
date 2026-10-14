"""Validate actual JSONL data and RunPod hardware before loading model weights."""
import argparse
import json
import shutil
from pathlib import Path


def validate_file(path):
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            for key in ("input", "prompt"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{number}: invalid {key}")
            answers = row.get("answer")
            answers = [answers] if isinstance(answers, str) else answers
            if not isinstance(answers, list) or not answers or not all(
                isinstance(answer, str) and answer.strip() for answer in answers
            ):
                raise ValueError(f"{path}:{number}: invalid answer")
            count += 1
    if not count:
        raise ValueError(f"Empty dataset: {path}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in (args.train_file, args.eval_file):
        print(path, validate_file(path), "valid rows")
    import torch
    import peft
    import transformers
    from check_cuda import check_cuda
    check_cuda(args.num_gpus)
    output = Path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Free disk GiB:", round(shutil.disk_usage(output.parent).free / 2**30, 1))
    print("Versions:", torch.__version__, transformers.__version__, peft.__version__)


if __name__ == "__main__":
    main()
