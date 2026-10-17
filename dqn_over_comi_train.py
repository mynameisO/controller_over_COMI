"""Train only the DQN controller, using a fixed COMI model for responses.

Example (run from this directory):
    python dqn_over_comi_train.py --model_name_or_path /path/to/base_model \
        --restore_from /path/to/comi_checkpoint --train_file data/train.jsonl \
        --output_dir output/dqn --dqn_merge_sizes 16,32 --dqn_epochs 5

Uses one process on one GPU (or CPU). Existing controller checkpoints are
loaded with their actions, encoder, optimizer, replay buffer, and step counts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dqn_prompt_controller.run_dqn_over_comi import load_samples, write_jsonl

HERE = Path(__file__).resolve().parent


@dataclass
class DQNTrainingArguments:
    dqn_comi_dir: str = field(default=str(HERE), metadata={'help': 'folder containing modeling_comi.py'})
    dqn_checkpoint: str = field(default='', metadata={'help': 'load if present and save here; default: output_dir/dqn_controller.pt'})
    dqn_merge_sizes: str = field(default='', metadata={'help': 'action set for a new controller; empty = COMI merge_sizes'})
    dqn_encoder_mode: str = field(default='hf', metadata={'help': 'hf | hash (hash is for pipeline tests)'})
    dqn_encoder_model: str = field(default='sentence-transformers/all-MiniLM-L6-v2')
    dqn_epochs: int = field(default=5, metadata={'help': 'epochs to run in this invocation'})
    dqn_epsilon_decay_steps: int = field(default=0, metadata={'help': '0 = 70% of train_size * epochs'})
    dqn_batch_size: int = field(default=1, metadata={'help': 'samples per controller batch'})
    dqn_num_train: int = field(default=0, metadata={'help': '0 = all training rows'})


def prepare_model(model_args, training_args, device):
    """Load COMI for generation and freeze all of its parameters."""
    from peft import LoraConfig
    from modeling_comi import COMI, build_lora_config, load_checkpoint_file

    model_args.train = False
    comi = COMI(model_args, training_args, lora_config=LoraConfig(**build_lora_config(model_args)))
    if training_args.restore_from:
        state_dict, checkpoint = load_checkpoint_file(training_args.restore_from)
        print(f'Loading COMI checkpoint {checkpoint}')
        comi.load_state_dict(state_dict, strict=False)
    comi.requires_grad_(False)
    comi.to(device).eval()
    return comi


def prepare_dqn_model(model_args, training_args, dqn_args, device, train_size):
    """Resume a controller or create one with the reference action/decay setup."""
    from prompt_controller import (
        DQNConfig, EncoderConfig, PromptController, TokenEncoder, actions_from_merge_sizes,
    )
    from comi_dqn import DQNOverCOMI

    checkpoint = (Path(dqn_args.dqn_checkpoint) if dqn_args.dqn_checkpoint
                  else Path(training_args.output_dir) / 'dqn_controller.pt')
    if checkpoint.exists():
        controller = PromptController.load(checkpoint, device=str(device))
        print(f'Loaded controller {checkpoint} (env_steps={controller.agent.env_steps})')
    else:
        sizes = dqn_args.dqn_merge_sizes or model_args.merge_sizes
        decay = dqn_args.dqn_epsilon_decay_steps or max(1, int(0.7 * train_size * dqn_args.dqn_epochs))
        controller = PromptController(
            TokenEncoder(EncoderConfig(dqn_args.dqn_encoder_mode, dqn_args.dqn_encoder_model),
                         device=str(device)),
            actions_from_merge_sizes(sizes), DQNConfig(epsilon_decay_steps=decay),
            device=str(device),
        )
        print(f'New controller: merge sizes {sizes}, epsilon decay {decay} steps')
    comi = prepare_model(model_args, training_args, device)
    return DQNOverCOMI(comi, controller, device=device), checkpoint


def train_dqn(runner, samples, model_args, training_args, dqn_args, checkpoint):
    """Generate COMI responses, feed rewards to DQN, and save each epoch."""
    from comi_dqn import summarize

    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    controller = runner.controller
    config = {
        'dqn_args': asdict(dqn_args),
        'merge_sizes': [int(round(a.compress_rate)) for a in controller.actions],
        'train_size': len(samples),
        'dqn_config': asdict(controller.agent.config),
        'reward_config': asdict(controller.reward_calc.config),
        'encoder': asdict(controller.encoder.config),
        'comi_model': model_args.model_name_or_path,
        'comi_restore_from': training_args.restore_from,
    }
    (out_dir / 'run_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    history = []
    # Do not wrap this loop in inference_mode: feedback must backpropagate
    # through the DQN. DQNOverCOMI._generate already guards COMI inference.
    for epoch in range(1, dqn_args.dqn_epochs + 1):
        random.shuffle(samples)
        rows = runner.run(samples, explore=True, train=True, batch_size=dqn_args.dqn_batch_size)
        losses = [r['loss'] for r in rows if r.get('loss') is not None]
        record = {
            'epoch': epoch, **summarize(rows),
            'loss_mean': sum(losses) / len(losses) if losses else None,
            'epsilon': controller.agent.epsilon(),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        controller.save(checkpoint)
        write_jsonl(out_dir / f'train_epoch{epoch}.jsonl', rows)
        (out_dir / 'training_history.json').write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved controller to {checkpoint}')
    return history


def main():
    # Independent DQN replicas do not synchronize replay or policy updates.
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise SystemExit('DQN training supports one process; launch with python, not multi-process torchrun')
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dqn_comi_dir', default=str(HERE))
    pre_args, _ = pre_parser.parse_known_args()
    sys.path.insert(0, str(Path(pre_args.dqn_comi_dir).resolve()))

    import torch
    from transformers import HfArgumentParser
    from modeling_comi import DataArguments, ModelArguments, TrainingArguments
    from prompt_controller import set_seed

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DQNTrainingArguments))
    model_args, data_args, training_args, dqn_args = parser.parse_args_into_dataclasses()
    if dqn_args.dqn_epochs <= 0 or dqn_args.dqn_batch_size <= 0:
        raise ValueError('dqn_epochs and dqn_batch_size must be positive')
    if dqn_args.dqn_num_train < 0 or dqn_args.dqn_epsilon_decay_steps < 0:
        raise ValueError('dqn_num_train and dqn_epsilon_decay_steps must be nonnegative')
    set_seed(training_args.seed)
    samples = load_samples(data_args.train_file, dqn_args.dqn_num_train, True, training_args.seed)
    if not samples:
        raise ValueError('Training data contains no samples')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runner, checkpoint = prepare_dqn_model(model_args, training_args, dqn_args, device, len(samples))
    train_dqn(runner, samples, model_args, training_args, dqn_args, checkpoint)


if __name__ == '__main__':
    main()
