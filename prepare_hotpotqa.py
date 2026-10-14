"""Convert HotpotQA to COMI JSONL without selecting gold supporting facts."""
import argparse
import hashlib
import json
from pathlib import Path


def convert_row(row):
    context = row["context"]
    titles, paragraphs = context["title"], context["sentences"]
    if len(titles) != len(paragraphs) or not titles:
        raise ValueError(f"Invalid context for {row['id']}")
    return {
        "input": "\n\n".join(
            title + "\n" + " ".join(sentence.strip() for sentence in sentences)
            for title, sentences in zip(titles, paragraphs)
        ),
        "prompt": row["question"],
        "answer": [row["answer"]],
        "id": row["id"],
    }


def export_split(rows, path):
    count = 0
    digest = hashlib.sha256()
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            converted = convert_row(row)
            if not all(converted[key].strip() for key in ("input", "prompt")):
                raise ValueError(f"Empty text for {row['id']}")
            if not converted["answer"][0].strip():
                raise ValueError(f"Empty answer for {row['id']}")
            line = (json.dumps(converted, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            count += 1
    if count == 0:
        raise ValueError(f"No rows for {path}")
    temporary.replace(path)
    return {"rows": count, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/hotpotqa")
    parser.add_argument("--config", choices=["distractor", "fullwiki"], default="distractor")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if not args.force and any(output.glob("*.json*")):
        raise SystemExit("Output already contains data. Use a new directory or --force.")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    # Resolve main to a commit so the manifest records an immutable source.
    revision = HfApi().dataset_info("hotpotqa/hotpot_qa", revision=args.revision).sha
    dataset = load_dataset("hotpotqa/hotpot_qa", args.config, revision=revision)
    validation = dataset["validation"]
    if not 0 < args.eval_size <= len(validation):
        raise ValueError("eval-size must be positive and no larger than validation")
    splits = {
        "train": dataset["train"],
        "validation": validation,
        "eval": validation.shuffle(seed=args.seed).select(range(args.eval_size)),
        "smoke_train": dataset["train"].select(range(min(32, len(dataset["train"])))),
        "smoke_eval": validation.select(range(min(10, len(validation)))),
    }
    manifest = {
        "dataset": "hotpotqa/hotpot_qa", "config": args.config,
        "revision": revision, "seed": args.seed,
        "license": "CC-BY-SA-4.0",
        "context_policy": "all supplied titles and sentences; no gold evidence filtering",
        "eval_note": "eval and smoke_eval are subsets of validation, not independent test sets",
        "files": {},
    }
    for name, rows in splits.items():
        manifest["files"][f"{name}.jsonl"] = export_split(rows, output / f"{name}.jsonl")
        print(name, manifest["files"][f"{name}.jsonl"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
