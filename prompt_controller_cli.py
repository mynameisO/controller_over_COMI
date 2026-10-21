"""Optional JSONL CLI for the online COMI controller."""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from prompt_controller import (
    DEFAULT_ACTIONS, ActionSpec, DQNConfig, EncoderConfig, PromptController,
    RewardConfig, TokenEncoder, actions_from_merge_sizes, parse_actions, set_seed,
)


def read_jsonl(path) -> Iterable[Dict]:
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row.setdefault('request_id', str(i))
                yield row


def write_jsonl(path, rows: Iterable[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _actions_from_args(args) -> Tuple[ActionSpec, ...]:
    if args.merge_sizes and args.actions:
        raise SystemExit('use either --actions or --merge-sizes, not both')
    if args.merge_sizes:
        return actions_from_merge_sizes(args.merge_sizes)
    if args.actions:
        return parse_actions(args.actions)
    return DEFAULT_ACTIONS


def cmd_init(args):
    ctl = PromptController(
        TokenEncoder(EncoderConfig(args.encoder_mode, args.encoder_model)),
        _actions_from_args(args),
        DQNConfig(epsilon_decay_steps=args.epsilon_decay_steps),
        RewardConfig(cost_source=args.cost_source),
    )
    ctl.save(args.checkpoint)
    print(json.dumps({'checkpoint': args.checkpoint,
                      'actions': [[a.name, a.retention] for a in ctl.actions]}, ensure_ascii=False))


def cmd_decide(args):
    ctl = PromptController.load(args.checkpoint)
    out = []
    for row in read_jsonl(args.input):
        d = ctl.decide(str(row['context']), str(row['query']), row['request_id'], explore=args.explore)
        out.append(d.to_dict())
    write_jsonl(args.output, out)
    if args.explore:
        ctl.save(args.checkpoint)
    counts = Counter(d['action_name'] for d in out)
    print(json.dumps({'decisions': len(out), 'action_counts': counts, 'output': args.output}, ensure_ascii=False))


def cmd_train(args):
    ctl = PromptController.load(args.checkpoint)
    rewards, losses = [], []
    for path in args.feedback:
        for row in read_jsonl(path):
            result = ctl.feedback(row, float(row['quality']), int(row['compressed_tokens']),
                                  row.get('original_tokens'), row.get('latency_saving'),
                                  row.get('energy_saving'))
            rewards.append(result['reward'])
            losses.append(result['loss'])
    ctl.save(args.output or args.checkpoint)
    print(json.dumps({'feedback_rows': len(rewards),
                      'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
                      'updates_done': len(losses),
                      'loss_mean': float(np.mean(losses)) if losses else None,
                      'epsilon': ctl.agent.epsilon()}, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(description='Standalone DQN prompt controller for COMI')
    p.add_argument('--seed', type=int, default=42)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('init', help='create a new controller checkpoint')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--encoder-mode', choices=['hash', 'hf'], default='hash')
    s.add_argument('--encoder-model', default='sentence-transformers/all-MiniLM-L6-v2')
    s.add_argument('--actions', help='NAME:RETENTION,... (default NONE:1,LIGHT:0.75,MEDIUM:0.5,STRONG:0.25)')
    s.add_argument('--merge-sizes', help='COMI merge sizes e.g. 1,4,16,32 (retention = 1/size)')
    s.add_argument('--epsilon-decay-steps', type=int, default=1000)
    s.add_argument('--cost-source', choices=['tokens', 'measured'], default='tokens')
    s.set_defaults(func=cmd_init)

    s = sub.add_parser('decide', help='requests.jsonl -> decisions.jsonl for COMI')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--input', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--explore', action='store_true', help='epsilon-greedy (use while collecting training data)')
    s.set_defaults(func=cmd_decide)

    s = sub.add_parser('train', help='COMI feedback.jsonl -> updated checkpoint')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--feedback', required=True, nargs='+')
    s.add_argument('--output', help='save to another path (default overwrite --checkpoint)')
    s.set_defaults(func=cmd_train)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    args.func(args)


if __name__ == '__main__':
    main()
