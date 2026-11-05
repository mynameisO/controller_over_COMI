"""Compression policy, Q-network, rewards, updates, and controller checkpoints.

Each request is a contextual bandit step: train Q(state, action) directly on
its reward, without replay or a target network. Checkpoint keys and network
parameter names preserve compatibility with existing controller checkpoints.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from inference_utility import (
    FEATURE_VERSION, STATE_NAMES, EncoderConfig, StateBuilder, StateFeatures,
    TokenEncoder, cosine_matrix, set_seed, simple_tokenize,
)


@dataclass(frozen=True)
class ActionSpec:
    """Compression choice: compress_rate = 1 / retention."""
    name: str
    retention: float

    @property
    def compress_rate(self) -> float:
        return 1.0 / self.retention


DEFAULT_ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec('NONE', 1.00),
    ActionSpec('LIGHT', 0.75),
    ActionSpec('MEDIUM', 0.50),
    ActionSpec('STRONG', 0.25),
)


def validate_actions(actions: Sequence[ActionSpec]) -> Tuple[ActionSpec, ...]:
    actions = tuple(actions)
    if not actions:
        raise ValueError('actions must not be empty')
    names = [a.name for a in actions]
    if len(set(names)) != len(names):
        raise ValueError('action names must be unique; order defines action index')
    for a in actions:
        if not a.name or not (0 < a.retention <= 1) or not math.isfinite(a.retention):
            raise ValueError(f'invalid action {a}: retention must be in (0, 1]')
    return actions


def parse_actions(text: str) -> Tuple[ActionSpec, ...]:
    """'NONE:1.0,LIGHT:0.75' -> ActionSpec list"""
    out = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(':')
        if not value:
            raise ValueError(f'bad action "{part}", expected NAME:RETENTION')
        out.append(ActionSpec(name.strip(), float(value)))
    return validate_actions(out)


def actions_from_merge_sizes(text: str) -> Tuple[ActionSpec, ...]:
    """'1,4,16,32' -> merge_size ของ COMI (retention = 1/merge_size)"""
    sizes = [int(x) for x in text.split(',') if x.strip()]
    if any(s <= 0 for s in sizes):
        raise ValueError('merge sizes must be positive integers')
    return validate_actions([ActionSpec(f'MERGE_{s}', 1.0 / s) for s in sizes])


@dataclass
class DQNConfig:
    state_dim: int = 7
    action_dim: int = 4
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1000
    hidden_1: int = 64
    hidden_2: int = 64

@dataclass
class RewardConfig:
    quality_weight: float = 0.55
    token_weight: float = 0.20
    energy_weight: float = 0.15
    latency_weight: float = 0.10
    quality_min: float = 0.70
    quality_penalty: float = 1.0

    cost_source: str = 'tokens'

class RewardCalculator:
    """Weighted quality and savings, with a penalty below quality_min."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        if self.config.cost_source not in ('tokens', 'measured'):
            raise ValueError('cost_source must be tokens or measured')

    def calculate(self, quality: float, token_saving: float,
                  energy_saving: Optional[float] = None, latency_saving: Optional[float] = None) -> float:
        c = self.config
        quality, token_saving = float(quality), float(token_saving)
        if not (math.isfinite(quality) and 0 <= quality <= 1):
            raise ValueError('quality must be finite and in [0, 1]')
        if not (math.isfinite(token_saving) and -1 <= token_saving <= 1):
            raise ValueError('token_saving must be in [-1, 1]')
        if c.cost_source == 'tokens':
            energy_saving = latency_saving = token_saving
        elif energy_saving is None or latency_saving is None:
            raise ValueError('cost_source=measured needs energy_saving and latency_saving')
        energy_saving = float(np.clip(energy_saving, -1, 1))
        latency_saving = float(np.clip(latency_saving, -1, 1))
        reward = (c.quality_weight * quality + c.token_weight * token_saving
                  + c.energy_weight * energy_saving + c.latency_weight * latency_saving)
        if quality < c.quality_min:
            reward -= c.quality_penalty
        return float(reward)

    @staticmethod
    def token_saving(original_tokens: int, compressed_tokens: int) -> float:
        if original_tokens <= 0:
            raise ValueError('original_tokens must be > 0')
        if compressed_tokens < 0:
            raise ValueError('compressed_tokens must be >= 0')
        return float(np.clip(1.0 - compressed_tokens / original_tokens, -1, 1))

class QNetwork(nn.Module):
    def __init__(self, state_dim=7, action_dim=4, hidden=(64, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """Learn Q(state, action) directly from each request's reward."""

    def __init__(self, config: DQNConfig, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = QNetwork(config.state_dim, config.action_dim,
                                   (config.hidden_1, config.hidden_2)).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.total_steps = 0
        self.env_steps = 0

    def epsilon(self) -> float:
        c = self.config
        progress = min(1.0, self.env_steps / max(c.epsilon_decay_steps, 1))
        return c.epsilon_start + progress * (c.epsilon_end - c.epsilon_start)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.policy_net(state).cpu().numpy()

    def update(self, state, action: int, reward: float) -> float:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        prediction = self.policy_net(state)[action]
        target = torch.tensor(reward, dtype=torch.float32, device=self.device)
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        self.total_steps += 1
        return loss.item()

@dataclass
class ControllerDecision:
    """COMI compression choice and the metadata needed for feedback."""
    request_id: str
    action_index: int
    action_name: str
    retention_ratio: float
    compress_rate: float
    input_tokens: int
    target_tokens: int
    state: List[float]
    q_values: List[float]
    explored: bool
    epsilon: Optional[float]
    feature_version: str = FEATURE_VERSION
    merge_size: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)

class PromptController:
    """Choose compression, learn from feedback, and save/load checkpoints."""

    def __init__(self, encoder: Optional[TokenEncoder] = None,
                 actions: Sequence[ActionSpec] = DEFAULT_ACTIONS,
                 dqn_config: Optional[DQNConfig] = None,
                 reward_config: Optional[RewardConfig] = None,
                 max_pairwise: int = 512, normalize_length_by: float = 4096.0, device=None):
        self.actions = validate_actions(actions)
        self.encoder = encoder or TokenEncoder(EncoderConfig(), device=device)
        self.state_builder = StateBuilder(self.encoder, max_pairwise, normalize_length_by)
        cfg = dqn_config or DQNConfig()
        cfg.action_dim = len(self.actions)
        if cfg.state_dim != len(STATE_NAMES):
            raise ValueError(f'state_dim must be {len(STATE_NAMES)}')
        self.agent = DQNAgent(cfg, device=device)
        self.reward_calc = RewardCalculator(reward_config)

    def build_state(self, context: str, query: str) -> Tuple[np.ndarray, int]:
        state, n = self.state_builder.build(context, query)
        return state.to_numpy(), n

    def decide(self, context: str, query: str, request_id: Optional[str] = None,
               explore: bool = False) -> ControllerDecision:
        """explore=True ตอน train (สุ่มตาม epsilon), False ตอน evaluate (เลือก Q สูงสุดเสมอ)"""
        state, n = self.build_state(context, query)
        return self.decide_from_state(state, n, request_id, explore)

    def decide_from_state(self, state: np.ndarray, input_tokens: int,
                          request_id: Optional[str] = None, explore: bool = False) -> ControllerDecision:
        state = np.asarray(state, dtype=np.float32)
        eps = self.agent.epsilon() if explore else None
        q_values = self.agent.q_values(state)
        explored = explore and random.random() < eps
        action = random.randrange(len(self.actions)) if explored else int(np.argmax(q_values))
        if explore:
            self.agent.env_steps += 1
        spec = self.actions[action]

        merge = round(spec.compress_rate)
        return ControllerDecision(
            request_id=str(request_id) if request_id is not None else '',
            action_index=action, action_name=spec.name,
            retention_ratio=spec.retention, compress_rate=spec.compress_rate,
            input_tokens=int(input_tokens),
            target_tokens=max(1, math.ceil(input_tokens * spec.retention)) if input_tokens else 0,
            state=state.tolist(), q_values=q_values.tolist(),
            explored=explored, epsilon=eps,
            merge_size=merge if abs(merge - spec.compress_rate) < 1e-6 else None,
        )

    def _check_decision(self, d: Dict):
        if d.get('feature_version', FEATURE_VERSION) != FEATURE_VERSION:
            raise ValueError('decision was made with a different feature definition')
        i = int(d['action_index'])
        if not 0 <= i < len(self.actions) or self.actions[i].name != d.get('action_name', self.actions[i].name):
            raise ValueError(f'action {d.get("action_name")}/{i} does not match this controller')
        if len(d['state']) != len(STATE_NAMES):
            raise ValueError('state must have 7 values')

    def feedback(self, decision, quality: float, compressed_tokens: int,
                 original_tokens: Optional[int] = None, latency_saving: Optional[float] = None,
                 energy_saving: Optional[float] = None, train: bool = True) -> Dict:
        """Calculate reward and update once. train=False calculates metrics only."""
        d = decision.to_dict() if isinstance(decision, ControllerDecision) else dict(decision)
        self._check_decision(d)
        original = int(original_tokens if original_tokens is not None else d['input_tokens'])
        saving = self.reward_calc.token_saving(original, int(compressed_tokens))
        reward = self.reward_calc.calculate(quality, saving, energy_saving, latency_saving)
        loss = self.agent.update(d['state'], int(d['action_index']), reward) if train else None
        return {'reward': reward, 'token_saving': saving, 'loss': loss}

    def save(self, path):
        a = self.agent
        payload = {
            'feature_version': FEATURE_VERSION,
            'actions': [[s.name, s.retention] for s in self.actions],
            'encoder': asdict(self.encoder.config),
            'max_pairwise': self.state_builder.max_pairwise,
            'normalize_length_by': self.state_builder.normalize_length_by,
            'dqn_config': asdict(a.config),
            'reward_config': asdict(self.reward_calc.config),
            'policy': a.policy_net.state_dict(),
            'optimizer': a.optimizer.state_dict(),
            'total_steps': a.total_steps, 'env_steps': a.env_steps,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, encoder: Optional[TokenEncoder] = None, device=None) -> 'PromptController':
        p = torch.load(path, map_location=device or 'cpu', weights_only=True)
        if p.get('feature_version') != FEATURE_VERSION:
            raise ValueError('checkpoint uses a different feature definition')
        enc_cfg = EncoderConfig(**p['encoder'])
        if encoder is None:
            encoder = TokenEncoder(enc_cfg, device=device)
        elif asdict(encoder.config) != asdict(enc_cfg):
            raise ValueError('encoder config does not match checkpoint')
        config = {f.name: p['dqn_config'][f.name] for f in fields(DQNConfig)}
        ctl = cls(encoder, [ActionSpec(n, float(r)) for n, r in p['actions']],
                  DQNConfig(**config), RewardConfig(**p['reward_config']),
                  p['max_pairwise'], p['normalize_length_by'], device=device)
        a = ctl.agent
        a.policy_net.load_state_dict(p['policy'])
        a.optimizer.load_state_dict(p['optimizer'])
        a.total_steps, a.env_steps = int(p['total_steps']), int(p['env_steps'])
        return ctl
