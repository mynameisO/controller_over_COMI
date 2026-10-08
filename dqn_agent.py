"""DQN extracted from dqn_prompt_compression.py without algorithm changes.

No tokenizer, encoder, compressor, language model, reward formula, or dataset.
The defaults retain the source's single-decision setup (gamma=0).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DQNConfig:
    """Hyperparameters only; state and action meanings belong to the caller."""
    state_dim: int = 7
    action_dim: int = 4
    learning_rate: float = 0.001
    gamma: float = 0.0
    batch_size: int = 32
    replay_capacity: int = 5000
    warmup_steps: int = 32
    target_update_every: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1000
    hidden_1: int = 64
    hidden_2: int = 64


@dataclass
class Transition:
    """One recorded interaction: (state, action, reward, next_state, done)."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Bounded experience history; random minibatches are sampled without replacement."""

    def __init__(self, capacity=5000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """MLP that maps a numeric state to one estimated return per discrete action."""

    def __init__(self, state_dim=7, action_dim=4, hidden=(64, 64)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], action_dim))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Environment-independent DQN. Depends only on NumPy and PyTorch."""

    def __init__(self, config: DQNConfig, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = QNetwork(config.state_dim, config.action_dim, (config.hidden_1, config.hidden_2)).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, (config.hidden_1, config.hidden_2)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.total_steps = 0
        self.env_steps = 0

    def epsilon(self):
        """Linear exploration schedule driven by training action selections."""
        c = self.config
        f = min(1.0, self.env_steps / max(c.epsilon_decay_steps, 1))
        return c.epsilon_start + f * (c.epsilon_end - c.epsilon_start)

    def select_action(self, state, training=True):
        """Return an integer action index. Evaluation is greedy and does not advance exploration."""
        if training:
            eps = self.epsilon()
            self.env_steps += 1
            if random.random() < eps:
                return random.randrange(self.config.action_dim)
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(x)
        return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store an experience. Pass NumPy vectors of shape (config.state_dim,)."""
        self.replay.push(Transition(state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done)))

    def train_step(self):
        """One replay update; return Huber loss, or None until the buffer reaches warmup."""
        c = self.config
        if len(self.replay) < max(c.batch_size, c.warmup_steps):
            return None
        b = self.replay.sample(c.batch_size)
        states = torch.tensor(np.stack([x.state for x in b]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([x.action for x in b], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([x.reward for x in b], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack([x.next_state for x in b]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([x.done for x in b], dtype=torch.float32, device=self.device).unsqueeze(1)
        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + c.gamma * next_q * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        self.total_steps += 1
        if self.total_steps % c.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, path):
        """Save networks, optimizer, config and counters. Replay and RNG states are not saved."""
        torch.save({'config': asdict(self.config), 'policy': self.policy_net.state_dict(), 'target': self.target_net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'steps': self.total_steps, 'env_steps': self.env_steps}, path)

    def load(self, path):
        """Restore into an agent created with the same config. Replay is not restored."""
        p = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(p['policy'])
        self.target_net.load_state_dict(p['target'])
        self.optimizer.load_state_dict(p['optimizer'])
        self.total_steps = int(p.get('steps', 0))
        self.env_steps = int(p.get('env_steps', 0))
