import json
import math
import os
import random
from pathlib import Path
import torch
import torch.distributed as dist
from tqdm import tqdm
from eval_utils import compute_exact, compute_f1, normalize_answer
from dataclasses import dataclass, field

HERE = Path(__file__).resolve().parent

@dataclass
class DQNArguments:
    dqn_mode: str = field(default='eval', metadata={'help': 'eval only; use run_dqn_over_comi.py for training'})
    dqn_checkpoint: str = field(default='', metadata={'help': 'controller checkpoint; train: resume if exists'})
    dqn_comi_dir: str = field(default=str(HERE.parent), metadata={'help': 'folder containing modeling_comi.py'})
    dqn_merge_sizes: str = field(default='', metadata={'help': 'action set; empty = COMI --merge_sizes'})
    dqn_encoder_mode: str = field(default='hf', metadata={'help': 'hf | hash (hash = pipeline test only)'})
    dqn_encoder_model: str = field(default='sentence-transformers/all-MiniLM-L6-v2')
    dqn_epochs: int = field(default=5)
    dqn_epsilon_decay_steps: int = field(default=0, metadata={'help': '0 = 70% of train_size * epochs'})
    dqn_batch_size: int = field(default=1, metadata={'help': 'samples per controller step'})
    dqn_num_train: int = field(default=0, metadata={'help': '0 = all train rows'})
    dqn_compare_fixed: bool = field(default=True, metadata={'help': 'eval: also run every fixed merge_size'})

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return distributed, rank, world_size, local_rank, device


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def load_eval_samples(test_file, num_samples, shuffle_samples, seed):
    samples = []
    with open(test_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            missing = {"input", "prompt", "answer"} - sample.keys()
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"Missing fields at line {line_number}: {names}")
            samples.append(sample)
    if shuffle_samples:
        random.Random(seed).shuffle(samples)
    return samples if num_samples <= 0 else samples[:num_samples]


def make_output_paths(output_dir, num_samples, merge_size):
    eval_dir = Path(output_dir) / "eval_result"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = "all" if num_samples == 0 else str(num_samples)
    output_path = eval_dir / f"nq_inference_results_{sample_tag}_{merge_size}.jsonl"
    metrics_path = eval_dir / f"nq_inference_metrics_{sample_tag}_{merge_size}.json"
    tmp_dir = eval_dir / f".tmp_rank_outputs_{sample_tag}_{merge_size}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return output_path, metrics_path, tmp_dir


def prepare_model(model_args, training_args, device):
    from peft import LoraConfig
    from modeling_comi import COMI, build_lora_config, load_checkpoint_file

    model_args.train = False
    lora_config = LoraConfig(**build_lora_config(model_args))
    model = COMI(model_args, training_args, lora_config=lora_config)
    if training_args.restore_from:
        state_dict, ckpt_file = load_checkpoint_file(training_args.restore_from)
        print(f"Loading checkpoint from {ckpt_file}")
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def prepare_dqn_model(model_args, training_args, dqn_args, device):
    """Load a trained controller and COMI, returning their inference wrapper."""
    from prompt_controller import PromptController
    from comi_dqn import DQNOverCOMI

    if dqn_args.dqn_mode != 'eval':
        raise ValueError('DQN inference requires --dqn_mode eval; use run_dqn_over_comi.py for training')
    checkpoint = (
        Path(dqn_args.dqn_checkpoint)
        if dqn_args.dqn_checkpoint
        else Path(training_args.output_dir) / 'dqn_controller.pt'
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Controller checkpoint not found: {checkpoint}')

    controller = PromptController.load(checkpoint, device=str(device))
    print(f'Loaded controller {checkpoint} (env_steps={controller.agent.env_steps})')
    comi = prepare_model(model_args, training_args, device)
    return DQNOverCOMI(comi, controller, device=device)


def infer_rank_batches(model, samples, batch_size, training_args, device, rank, world_size):
    rank_samples = samples[rank::world_size]
    generated_rows = []
    iterator = chunked(rank_samples, batch_size)
    progress = tqdm(iterator, total=math.ceil(len(rank_samples) / batch_size), disable=rank != 0, desc="Inference")

    for batch in progress:
        rows = model.run(batch, explore=False, train=False, batch_size=batch_size)
        for sample, row in zip(batch, rows):
            row.update(context=sample.get("input", ""), question=sample.get("prompt", ""))
            generated_rows.append(row)

    return generated_rows


def write_rank_results(rank_rows, tmp_dir, rank):
    rank_file = tmp_dir / f"rank_{rank:05d}.jsonl"
    with open(rank_file, "w", encoding="utf-8") as f:
        for row in rank_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rank_file


def merge_rank_outputs(tmp_dir, world_size, final_output_path):
    merged_rows = []
    for rank in range(world_size):
        rank_file = tmp_dir / f"rank_{rank:05d}.jsonl"
        if not rank_file.exists():
            continue
        with open(rank_file, "r", encoding="utf-8") as f:
            for line in f:
                merged_rows.append(json.loads(line))

    merged_rows.sort(key=lambda row: row.get("sample_id", 0))
    with open(final_output_path, "w", encoding="utf-8") as out_f:
        for row in merged_rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return merged_rows


def compute_metrics(rows):
    em_scores = []
    f1_scores = []

    for item in rows:
        prediction = item["model_output"]
        ground_truth = item["ground_truth"]

        if isinstance(ground_truth, list):
            em_score = max(compute_exact(normalize_answer(gt), normalize_answer(prediction)) for gt in ground_truth)
            f1_score = max(compute_f1(normalize_answer(prediction), normalize_answer(gt)) for gt in ground_truth)
        else:
            em_score = compute_exact(normalize_answer(ground_truth), normalize_answer(prediction))
            f1_score = compute_f1(normalize_answer(prediction), normalize_answer(ground_truth))

        em_scores.append(em_score)
        f1_scores.append(f1_score)

    return {
        "total_samples": len(em_scores),
        "avg_em": sum(em_scores) / len(em_scores) if em_scores else 0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
    }


def main():
    from transformers import HfArgumentParser

    from modeling_comi import DataArguments, ModelArguments, TrainingArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DQNArguments))
    model_args, data_args, training_args, dqn_args = parser.parse_args_into_dataclasses()

    distributed, rank, world_size, local_rank, device = setup_distributed()
    random.seed(training_args.seed + rank)
    torch.manual_seed(training_args.seed + rank)

    output_path, metrics_path, tmp_dir = make_output_paths(training_args.output_dir, data_args.num_samples, "dqn")

    if rank == 0:
        print(f"Initializing model from {model_args.model_name_or_path}")
    model = prepare_dqn_model(model_args, training_args, dqn_args, device)

    samples = load_eval_samples(
        data_args.test_file,
        num_samples=data_args.num_samples,
        shuffle_samples=data_args.shuffle_samples,
        seed=training_args.seed,
    )
    for idx, sample in enumerate(samples):
        sample["sample_id"] = idx

    if rank == 0:
        print(
            f"Running inference on {len(samples)} samples with world_size={world_size}, "
            f"per_device_eval_batch_size={training_args.per_device_eval_batch_size}"
        )

    rank_rows = infer_rank_batches(
        model=model,
        samples=samples,
        batch_size=training_args.per_device_eval_batch_size,
        training_args=training_args,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    write_rank_results(rank_rows, tmp_dir, rank)

    barrier()

    if rank == 0:
        merged_rows = merge_rank_outputs(tmp_dir, world_size, output_path)
        final_metrics = compute_metrics(merged_rows)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)

        print(f"Inference finished. Results saved to: {output_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(
            f"Final metrics: EM={final_metrics['avg_em']:.4f}, "
            f"F1={final_metrics['avg_f1']:.4f}"
        )

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
