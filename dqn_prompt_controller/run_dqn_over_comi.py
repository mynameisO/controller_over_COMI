"""Train / evaluate DQN ที่ครอบ COMI

วางโฟลเดอร์นี้ไว้ใน repo COMI (หรือชี้ --comi_dir) แล้วรันจากในโฟลเดอร์นี้
COMI argument ใช้ชื่อเดิมทั้งหมด (--model_name_or_path, --restore_from, --segment_size, ...)
ส่วน argument ของ DQN ขึ้นต้นด้วย --dqn_

Train:
  python run_dqn_over_comi.py --dqn_mode train \
      --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --restore_from /path/to/comi_ckpt \
      --train_file data/train.jsonl --output_dir runs/dqn --dqn_epochs 5

Eval (เทียบกับ fixed merge_size ทุกค่า):
  python run_dqn_over_comi.py --dqn_mode eval --dqn_checkpoint runs/dqn/dqn_controller.pt \
      --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --restore_from /path/to/comi_ckpt \
      --test_file data/test.jsonl --output_dir runs/dqn --dqn_compare_fixed True

ข้อมูลรูปแบบเดียวกับ ft_inference.py: {"input": context, "prompt": question, "answer": str | [str]}
รันได้ทีละ 1 process (ไม่รองรับ torchrun หลาย GPU)
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


# argument เพิ่มของ DQN (ใส่ใน command line เป็น --dqn_xxx)
# ตัวอย่างค่า:
#   --dqn_mode train --dqn_epochs 5          train DQN 5 รอบบน train_file
#   --dqn_merge_sizes 16,32                  DQN เลือกได้ 2 action: 16 หรือ 32
#   --dqn_epsilon_decay_steps 0              คิดอัตโนมัติ = 0.7 * จำนวน sample * epochs
#                                            เช่น 1000 sample * 5 epoch -> 3500 step
@dataclass
class DQNArguments:
    dqn_mode: str = field(default='train', metadata={'help': 'train | eval'})
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


def load_samples(path, limit=0, shuffle=False, seed=42):
    # อ่าน jsonl รูปแบบเดียวกับ ft_inference.py ทุกแถวต้องมี input / prompt / answer
    # limit = 0 คือใช้ทั้งหมด, ใส่ sample_id ให้ถ้ายังไม่มี (ไว้จับคู่ผลทีหลัง)
    rows = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                missing = {'input', 'prompt', 'answer'} - r.keys()
                if missing:
                    raise ValueError(f'{path}:{i} missing {sorted(missing)}')
                rows.append(r)
    if shuffle:
        random.Random(seed).shuffle(rows)
    rows = rows if limit <= 0 else rows[:limit]
    for i, r in enumerate(rows):
        r.setdefault('sample_id', i)
    return rows


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    # หา path ของ COMI ก่อน parse argument จริง เพราะต้อง import modeling_comi
    # เพื่อเอา ModelArguments / DataArguments / TrainingArguments ของ COMI มาใช้
    pre = [a for a in sys.argv[1:]]
    comi_dir = HERE.parent
    if '--dqn_comi_dir' in pre:
        comi_dir = Path(pre[pre.index('--dqn_comi_dir') + 1]).resolve()
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(comi_dir))

    import numpy as np
    import torch
    from peft import LoraConfig
    from transformers import HfArgumentParser
    from modeling_comi import (COMI, DataArguments, ModelArguments, TrainingArguments,
                               build_lora_config, load_checkpoint_file)
    from prompt_controller import (DQNConfig, EncoderConfig, PromptController, TokenEncoder,
                                   actions_from_merge_sizes, set_seed)
    from comi_dqn import DQNOverCOMI, summarize

    # parse argument ของ COMI + DQN พร้อมกัน (COMI ชื่อเดิม, DQN ขึ้นต้น dqn_)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DQNArguments))
    model_args, data_args, training_args, dqn_args = parser.parse_args_into_dataclasses()
    if dqn_args.dqn_mode not in ('train', 'eval'):
        raise SystemExit('--dqn_mode must be train or eval')
    set_seed(training_args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(dqn_args.dqn_checkpoint) if dqn_args.dqn_checkpoint else out_dir / 'dqn_controller.pt'

    # ---- COMI (โหลดแบบเดียวกับ ft_inference.prepare_model, ไม่ train COMI) ----
    model_args.train = False
    comi = COMI(model_args, training_args, lora_config=LoraConfig(**build_lora_config(model_args)))
    if training_args.restore_from:
        state_dict, ckpt_file = load_checkpoint_file(training_args.restore_from)
        print(f'Loading COMI checkpoint {ckpt_file}')
        comi.load_state_dict(state_dict, strict=False)
    comi.to(device).eval()

    # ---- data ----
    if dqn_args.dqn_mode == 'train':
        samples = load_samples(data_args.train_file, dqn_args.dqn_num_train, True, training_args.seed)
    else:
        samples = load_samples(data_args.test_file, data_args.num_samples, data_args.shuffle_samples,
                               training_args.seed)

    # ---- controller (standalone) ----
    # มี checkpoint แล้ว -> โหลดมาใช้/เทรนต่อ (action, weight, epsilon, replay มาครบ)
    # ยังไม่มี -> train: สร้างใหม่โดยใช้ merge_sizes เป็น action, eval: error
    if ckpt.exists():
        controller = PromptController.load(ckpt, device=str(device))
        print(f'Loaded controller {ckpt} (env_steps={controller.agent.env_steps})')
    elif dqn_args.dqn_mode == 'eval':
        raise SystemExit(f'controller checkpoint not found: {ckpt}')
    else:
        sizes = dqn_args.dqn_merge_sizes or model_args.merge_sizes
        decay = dqn_args.dqn_epsilon_decay_steps or max(1, int(0.7 * len(samples) * dqn_args.dqn_epochs))
        controller = PromptController(
            TokenEncoder(EncoderConfig(dqn_args.dqn_encoder_mode, dqn_args.dqn_encoder_model), device=str(device)),
            actions_from_merge_sizes(sizes), DQNConfig(epsilon_decay_steps=decay), device=str(device))
        print(f'New controller: merge sizes {sizes}, epsilon decay {decay} steps')

    runner = DQNOverCOMI(comi, controller, device=device)
    sizes = [int(round(a.compress_rate)) for a in controller.actions]

    # ---- train ----
    # ทุก epoch: สลับลำดับ sample -> DQN เลือก (มีสุ่มตาม epsilon) -> COMI ตอบ -> feedback -> train_step
    # จบแต่ละ epoch save checkpoint ทับ และเขียนผลรายตัวลง train_epoch{N}.jsonl
    if dqn_args.dqn_mode == 'train':
        history = []
        for epoch in range(1, dqn_args.dqn_epochs + 1):
            random.shuffle(samples)
            rows = runner.run(samples, explore=True, train=True, batch_size=dqn_args.dqn_batch_size)
            losses = [r['loss'] for r in rows if r.get('loss') is not None]
            rec = {'epoch': epoch, **summarize(rows),
                   'loss_mean': float(np.mean(losses)) if losses else None,
                   'epsilon': controller.agent.epsilon()}
            history.append(rec)
            print(json.dumps(rec, ensure_ascii=False))
            controller.save(ckpt)
            write_jsonl(out_dir / f'train_epoch{epoch}.jsonl', rows)
        (out_dir / 'training_history.json').write_text(json.dumps(history, indent=2, ensure_ascii=False))
        (out_dir / 'run_config.json').write_text(json.dumps({
            'dqn_args': asdict(dqn_args), 'merge_sizes': sizes, 'train_size': len(samples),
            'dqn_config': asdict(controller.agent.config),
            'reward_config': asdict(controller.reward_calc.config),
            'encoder': asdict(controller.encoder.config),
            'comi_model': model_args.model_name_or_path, 'comi_restore_from': training_args.restore_from,
        }, indent=2, ensure_ascii=False))
        print(f'Saved controller to {ckpt}')
        return

    # ---- eval ----
    # DQN เลือกแบบ greedy (ไม่สุ่ม, ไม่ train) แล้วรัน fixed merge_size ทุกค่าบนข้อมูลชุดเดียวกัน
    # ผลรวมอยู่ใน comparison.json เช่น {"DQN": {...}, "Fixed-16": {...}, "Fixed-32": {...}}
    results = {}
    rows = runner.run(samples, explore=False, train=False, batch_size=dqn_args.dqn_batch_size)
    write_jsonl(out_dir / 'eval_dqn.jsonl', rows)
    results['DQN'] = summarize(rows)
    if dqn_args.dqn_compare_fixed:
        for m in sizes:
            fixed = runner.run_fixed(samples, m, batch_size=training_args.per_device_eval_batch_size)
            write_jsonl(out_dir / f'eval_fixed_{m}.jsonl', fixed)
            results[f'Fixed-{m}'] = summarize(fixed)
    (out_dir / 'comparison.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for name, r in results.items():
        print(f"{name:10s} reward={r['reward']:.4f} F1={r['f1']:.4f} EM={r['em']:.4f} "
              f"saving={r['token_saving']:.4f} counts={r['merge_size_counts']}")


if __name__ == '__main__':
    main()
