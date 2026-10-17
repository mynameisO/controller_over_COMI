"""Offline CPU tests (hash encoder, no HF download, no LLM)

python tests/test_controller.py
"""
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompt_controller import (DEFAULT_ACTIONS, DQNConfig, PromptController, RewardCalculator,  # noqa: E402
                               StateBuilder, TokenEncoder, actions_from_merge_sizes, set_seed)

VOCAB = [f'w{i}' for i in range(5000)]


def make_request(kind, rng):
    """A = ซ้ำทั้งหมด (บีบหนักได้), C = ซ้ำครึ่งหนึ่ง, B = ไม่ซ้ำเลย (ห้ามบีบ)"""
    n = rng.randint(80, 200)
    if kind == 'A':
        base = rng.sample(VOCAB, 5)
        words = [rng.choice(base) for _ in range(n)]
    elif kind == 'C':
        uniq = rng.sample(VOCAB, n // 2)
        words = uniq + [rng.choice(uniq[:5]) for _ in range(n - n // 2)]
        rng.shuffle(words)
    else:
        words = rng.sample(VOCAB, n)
    return ' '.join(words), ' '.join(rng.sample(words, 3))


NEED = {'A': 0.25, 'C': 0.50, 'B': 1.00}   # retention ขั้นต่ำที่ยังตอบถูก


def fake_comi(kind, decision):
    """แทน COMI + LLM: quality ตกถ้าบีบเกินที่ข้อมูลรับได้"""
    quality = 1.0 if decision.retention_ratio >= NEED[kind] - 1e-9 else 0.3
    compressed = math.ceil(decision.input_tokens * decision.retention_ratio)
    return quality, compressed


def test_state_matches_bruteforce():
    enc = TokenEncoder()
    sb = StateBuilder(enc)
    ctx, q = 'a b c a d e f b', 'a c'
    state, n = sb.build(ctx, q)
    _, cv = enc.encode(ctx)
    _, qv = enc.encode(q)
    cn = torch.nn.functional.normalize(cv, dim=-1)
    qn = torch.nn.functional.normalize(qv.mean(0, keepdim=True), dim=-1)
    r = (cn @ qn.T).squeeze(1)
    sim = cn @ cn.T
    sim.fill_diagonal_(-1)
    d = sim.max(1).values
    m = r - d
    exp = [8 / 4096, r.mean(), r.max(), m.mean(), m.max(), m.std(unbiased=False), d.mean()]
    np.testing.assert_allclose(state.to_numpy(), np.array(exp, dtype=np.float32), atol=1e-6)
    assert n == 8
    # edge cases
    for ctx in ('', 'one', ' '.join(VOCAB[:1100])):
        s, _ = sb.build(ctx, 'query')
        assert s.to_numpy().shape == (7,) and np.isfinite(s.to_numpy()).all()


def test_reward_formula():
    rc = RewardCalculator()
    s = rc.token_saving(2048, 128)
    assert abs(s - 0.9375) < 1e-9
    assert abs(rc.calculate(0.8, s) - (0.55 * 0.8 + 0.45 * 0.9375)) < 1e-9
    assert abs(rc.calculate(0.65, 0.96875) - (0.55 * 0.65 + 0.45 * 0.96875 - 1.0)) < 1e-9
    for bad in ((-0.1, 0.5), (1.1, 0.5), (0.5, 2.0)):
        try:
            rc.calculate(*bad)
            raise AssertionError('should fail')
        except ValueError:
            pass


def test_learnability_and_checkpoint():
    set_seed(0)
    rng = random.Random(0)
    train = [(k, *make_request(k, rng)) for k in 'ABC' for _ in range(40)]
    test = [(k, *make_request(k, rng)) for k in 'ABC' for _ in range(20)]
    epochs = 15
    ctl = PromptController(dqn_config=DQNConfig(epsilon_decay_steps=int(0.7 * len(train) * epochs)), device='cpu')
    cache = {i: ctl.build_state(c, q) for i, (_, c, q) in enumerate(train)}
    for _ in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in order:
            kind = train[i][0]
            d = ctl.decide_from_state(*cache[i], request_id=str(i), explore=True)
            quality, comp = fake_comi(kind, d)
            ctl.feedback(d, quality, comp)

    def evaluate(policy):
        rewards = []
        for kind, c, q in test:
            d = policy(c, q)
            quality, comp = fake_comi(kind, d)
            rewards.append(ctl.reward_calc.calculate(quality, ctl.reward_calc.token_saving(d.input_tokens, comp)))
        return float(np.mean(rewards))

    dqn = evaluate(lambda c, q: ctl.decide(c, q))
    fixed = {}
    for idx, spec in enumerate(ctl.actions):
        def pol(c, q, idx=idx):
            d = ctl.decide(c, q)
            d.action_index, d.action_name, d.retention_ratio = idx, ctl.actions[idx].name, ctl.actions[idx].retention
            return d
        fixed[spec.name] = evaluate(pol)
    print('DQN', round(dqn, 4), 'fixed', {k: round(v, 4) for k, v in fixed.items()})
    assert all(dqn > v for v in fixed.values()), 'DQN must beat every fixed policy'
    picks = {k: ctl.decide(c, q).action_name for k, c, q in test[::20]}
    assert picks == {'A': 'STRONG', 'B': 'NONE', 'C': 'MEDIUM'}, picks

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'ctl.pt'
        ctl.save(path)
        loaded = PromptController.load(path)
        for _, c, q in test[:10]:
            assert loaded.decide(c, q).action_index == ctl.decide(c, q).action_index
        assert len(loaded.agent.replay) == len(ctl.agent.replay)
        assert loaded.agent.env_steps == ctl.agent.env_steps


def test_feedback_validation_and_merge_sizes():
    ctl = PromptController(actions=actions_from_merge_sizes('1,4,16,32'), device='cpu')
    d = ctl.decide('a b c d e f g h', 'a')
    assert d.merge_size in (1, 4, 16, 32) and d.compress_rate == d.merge_size
    other = PromptController(device='cpu')
    try:
        other.feedback(d, 1.0, 2)
        raise AssertionError('mismatched action table must fail')
    except ValueError:
        pass
    assert len(DEFAULT_ACTIONS) == 4 and other.agent.config.action_dim == 4


def test_cli_round_trip():
    rng = random.Random(1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reqs = []
        for i in range(40):
            kind = 'ABC'[i % 3]
            c, q = make_request(kind, rng)
            reqs.append({'request_id': f'r{i}', 'context': c, 'query': q, 'kind': kind})
        (tmp / 'req.jsonl').write_text('\n'.join(json.dumps(r) for r in reqs), encoding='utf-8')
        run = lambda *a: subprocess.run([sys.executable, str(ROOT / 'prompt_controller.py'), *a],
                                        check=True, capture_output=True, text=True).stdout
        ck = str(tmp / 'ctl.pt')
        run('init', '--checkpoint', ck)
        run('decide', '--checkpoint', ck, '--input', str(tmp / 'req.jsonl'),
            '--output', str(tmp / 'dec.jsonl'), '--explore')
        decs = [json.loads(l) for l in (tmp / 'dec.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [d['request_id'] for d in decs] == [r['request_id'] for r in reqs]
        kinds = {r['request_id']: r['kind'] for r in reqs}
        fb = []
        for d in decs:   # COMI ส่ง quality + compressed_tokens กลับมาพร้อม decision เดิม
            q = 1.0 if d['retention_ratio'] >= NEED[kinds[d['request_id']]] else 0.3
            fb.append({**d, 'quality': q, 'compressed_tokens': math.ceil(d['input_tokens'] * d['retention_ratio'])})
        (tmp / 'fb.jsonl').write_text('\n'.join(json.dumps(r) for r in fb), encoding='utf-8')
        out = json.loads(run('train', '--checkpoint', ck, '--feedback', str(tmp / 'fb.jsonl')))
        assert out['feedback_rows'] == 40 and out['updates_done'] == 40 and out['replay_size'] == 40
        ctl = PromptController.load(ck)
        assert ctl.agent.env_steps == 40 and ctl.agent.total_steps == 40


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
