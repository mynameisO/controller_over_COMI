"""DQNOverCOMI กับ COMI ปลอม (CPU, ไม่โหลด model)

python tests/test_comi_wrapper.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comi_dqn import DQNOverCOMI, squad_f1  # noqa: E402
from prompt_controller import DQNConfig, PromptController, actions_from_merge_sizes, set_seed  # noqa: E402


class FakeTokenizer:
    def __call__(self, texts, return_tensors='pt', padding=True, add_special_tokens=False):
        ids = [[hash(w) % 100 + 1 for w in t.split()] or [0] for t in texts]
        n = max(len(x) for x in ids)
        mask = [[1] * len(x) + [0] * (n - len(x)) for x in ids]
        ids = [x + [0] * (n - len(x)) for x in ids]
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask)}

    def batch_decode(self, out, skip_special_tokens=True):
        return out


class FakeCOMI(nn.Module):
    """interface เดียวกับ COMI: merge_size, is_random, build_comi_memory, forward(labels=None)
    ตอบถูกเฉพาะเมื่อ merge_size <= ค่าที่ซ่อนไว้ใน query"""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.tokenizer = FakeTokenizer()
        self.merge_size, self.is_random = 16, True
        self.calls = []

    def build_comi_memory(self, input_ids, attention_mask, query_ids, query_attention_mask):
        lens = attention_mask.sum(1)
        slots = [math.ceil(int(n) / self.merge_size) for n in lens]
        m = max(slots)
        mask = torch.tensor([[1] * s + [0] * (m - s) for s in slots], dtype=torch.bool)
        return torch.zeros(len(slots), m, 4), mask

    def forward(self, input_ids=None, attention_mask=None, query_ids=None, query_attention_mask=None, labels=None):
        assert labels is None and not self.is_random
        self.calls.append((self.merge_size, input_ids.shape[0]))
        self.build_comi_memory(input_ids, attention_mask, query_ids, query_attention_mask)
        return [f'ans{self.merge_size}' for _ in range(input_ids.shape[0])]


def make_samples(n):
    # ข้อความยาวและซ้ำมาก -> เลือก 16 ได้, ข้อความไม่ซ้ำ -> ต้อง 4
    out = []
    for i in range(n):
        if i % 2:
            ctx = ' '.join(['x', 'y'] * 60)
            ans = ['ans16', 'ans8', 'ans4']
        else:
            ctx = ' '.join(f'u{i}_{k}' for k in range(120))
            ans = 'ans4'
        out.append({'input': ctx, 'prompt': 'q', 'answer': ans, 'sample_id': i})
    return out


def main():
    set_seed(0)
    comi = FakeCOMI()
    ctl = PromptController(actions=actions_from_merge_sizes('4,8,16'),
                           dqn_config=DQNConfig(epsilon_decay_steps=200), device='cpu')
    runner = DQNOverCOMI(comi, ctl, device='cpu')

    rows = runner.run(make_samples(8), explore=True, train=True, batch_size=4)
    assert [r['sample_id'] for r in rows] == list(range(8)), 'order kept'
    assert comi.merge_size == 16 and comi.is_random is True, 'COMI settings restored'
    assert 'build_comi_memory' not in comi.__dict__, 'hook removed'
    for r in rows:
        assert r['compressed_tokens'] == math.ceil(r['original_tokens'] / r['merge_size'])
    assert sum(b for _, b in comi.calls) == 8 and all(b <= 4 for _, b in comi.calls)
    assert len(ctl.agent.replay) == 8

    train = make_samples(40)
    for _ in range(12):
        runner.run(train, explore=True, train=True, batch_size=8)
    test = make_samples(10)
    picks = {r['sample_id'] % 2: r['merge_size'] for r in runner.run(test)}
    assert picks == {0: 4, 1: 16}, picks
    dqn = sum(r['reward'] for r in runner.run(test)) / len(test)
    for m in (4, 8, 16):
        fixed = sum(r['reward'] for r in runner.run_fixed(test, m, batch_size=5)) / len(test)
        assert dqn > fixed, (m, dqn, fixed)
    print('DQN reward', round(dqn, 4))

    try:
        DQNOverCOMI(comi, PromptController(device='cpu'), device='cpu')
        raise AssertionError('non-integer merge size must be rejected')
    except ValueError:
        pass
    assert squad_f1('The Paris.', ['paris', 'x']) == 1.0
    print('PASS test_comi_wrapper')


if __name__ == '__main__':
    main()
