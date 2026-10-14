"""Standalone DQN controller สำหรับเลือกระดับการบีบอัด prompt

ไฟล์นี้แยกเฉพาะส่วน controller ออกมาจาก dqn_prompt_compression.py
- ไม่มี compressor และไม่เรียก LLM
- input  : context + query (text)
- output : decision (action, retention_ratio, compress_rate) ส่งต่อให้ COMI เป็นคนบีบ
- reward : COMI วัด quality และจำนวน token หลังบีบ แล้วส่งกลับมาให้ controller train

Flow
    Context + Query
    -> TokenEncoder (hash / HF encoder)
    -> relevance / redundancy / MIG
    -> State 7 ตัว
    -> DQN เลือก action
    -> decision ส่งให้ COMI
    -> COMI ส่ง quality + compressed_tokens กลับมา
    -> คำนวณ reward -> replay buffer -> train_step

State, QNetwork, DQNAgent, Reward ใช้สูตรเดียวกับไฟล์ต้นฉบับ (gamma=0, 1 decision ต่อ request)

อ่านไฟล์นี้ตามลำดับหัวข้อ
    1. Action       : DQN เลือกอะไรได้บ้าง (retention / merge_size)
    2. Config       : hyperparameter ของ DQN, reward, encoder
    3. State        : แปลง text -> vector -> feature 7 ตัว
    4. Reward       : เอาผลจาก COMI มาคิดเป็นคะแนน
    5. DQN          : Q-network, replay buffer, epsilon-greedy, train_step
    6. Controller   : class ที่คนอื่นเรียกใช้ (decide / feedback / save / load)
    7. CLI          : init / decide / train ผ่านไฟล์ jsonl

ตัวอย่างตัวเลขที่ใช้ใน comment ทั้งไฟล์: context ยาว 2048 คำ, เลือก merge_size 16
    -> COMI เหลือ 2048 / 16 = 128 slot -> token_saving = 1 - 128/2048 = 0.9375
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# FEATURE_VERSION ติดไปกับ checkpoint และ decision ทุกอัน
# ถ้าวันหลังแก้สูตร state ให้เปลี่ยนชื่อนี้ด้วย ไม่อย่างนั้นจะโหลด model เก่ามาใช้กับ state แบบใหม่โดยไม่รู้ตัว
FEATURE_VERSION = 'prompt-word-r-minus-d-v1'
# ลำดับของ state 7 ตัว (index 0..6) ต้องตรงกับ input ของ QNetwork เสมอ
STATE_NAMES = ('input_tokens', 'relevance_mean', 'relevance_max',
               'mig_mean', 'mig_max', 'mig_std', 'redundancy_mean')


def set_seed(seed: int = 42):
    # ล็อก random ทุกตัว (python, numpy, torch) ให้รันซ้ำได้ผลเดิม
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionSpec:
    """1 action = เก็บ context ไว้กี่ % (retention). compress_rate = 1 / retention

    ตัวอย่าง
        ActionSpec('MEDIUM', 0.5)     -> เก็บ 50%  -> compress_rate = 2
        ActionSpec('MERGE_16', 1/16)  -> เก็บ 6.25% -> compress_rate = 16 = merge_size ของ COMI
    ลำดับใน list = action index ที่ DQN ส่งออกมา (index 0 คือตัวแรก)
    """
    name: str        # ชื่อไว้อ่าน log และใช้เช็คว่า feedback ตรงกับ controller ตัวนี้
    retention: float  # สัดส่วนที่เก็บไว้ ต้องอยู่ในช่วง (0, 1]

    @property
    def compress_rate(self) -> float:
        return 1.0 / self.retention


# ค่าเดียวกับต้นฉบับ: NONE=100%, LIGHT=75%, MEDIUM=50%, STRONG=25%
DEFAULT_ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec('NONE', 1.00),
    ActionSpec('LIGHT', 0.75),
    ActionSpec('MEDIUM', 0.50),
    ActionSpec('STRONG', 0.25),
)


def validate_actions(actions: Sequence[ActionSpec]) -> Tuple[ActionSpec, ...]:
    # เช็คว่า action ไม่ว่าง, ชื่อไม่ซ้ำ, retention อยู่ใน (0, 1]
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    state_dim: int = 7              # จำนวน feature ของ state (ห้ามเปลี่ยน ถ้าไม่ได้แก้ StateBuilder)
    action_dim: int = 4             # จำนวน action, PromptController ตั้งให้อัตโนมัติ = len(actions)
    learning_rate: float = 1e-3     # lr ของ Adam
    gamma: float = 0.0              # 0 = ดู reward ของ request นี้อย่างเดียว (1 decision แล้วจบ)
    batch_size: int = 32            # จำนวน transition ที่สุ่มจาก replay ต่อ 1 update
    replay_capacity: int = 5000     # replay เก็บได้สูงสุดกี่อัน เต็มแล้วอันเก่าสุดหลุด
    warmup_steps: int = 32          # replay ต้องมีอย่างน้อยเท่านี้ก่อนเริ่ม train
    target_update_every: int = 100  # copy policy -> target ทุกกี่ update (gamma=0 ค่านี้ไม่มีผล)
    epsilon_start: float = 1.0      # ตอนเริ่ม สุ่ม action 100%
    epsilon_end: float = 0.05       # สุดท้าย สุ่ม 5% ที่เหลือเลือกตาม Q
    epsilon_decay_steps: int = 1000  # ลด epsilon แบบเส้นตรงจาก start -> end ภายในกี่ step
    hidden_1: int = 64              # ขนาด hidden layer 1 ของ QNetwork
    hidden_2: int = 64              # ขนาด hidden layer 2


@dataclass
class RewardConfig:
    # reward = 0.55*quality + 0.20*token + 0.15*energy + 0.10*latency  (ผลรวม weight = 1.0)
    quality_weight: float = 0.55    # น้ำหนักคุณภาพคำตอบ (F1)
    token_weight: float = 0.20      # น้ำหนัก token ที่ประหยัดได้
    energy_weight: float = 0.15     # น้ำหนักพลังงานที่ประหยัด
    latency_weight: float = 0.10    # น้ำหนักเวลาที่ประหยัด
    quality_min: float = 0.70       # ถ้า quality ต่ำกว่านี้ โดนหักคะแนน
    quality_penalty: float = 1.0    # หักเท่านี้ (ทำให้ reward ติดลบ -> DQN เลี่ยงการบีบจนคำตอบเสีย)
    # tokens   : latency/energy ใช้ token_saving แทน -> reward = 0.55q + 0.45s - penalty
    # measured : ใช้ latency_saving / energy_saving ที่ COMI วัดส่งมาจริง
    cost_source: str = 'tokens'


@dataclass
class EncoderConfig:
    mode: str = 'hash'                 # hash = ทดสอบ pipeline, hf = encoder จริง
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'  # ใช้เฉพาะ hf mode (HF id หรือ path ในเครื่อง)
    hash_dim: int = 128                # ขนาด vector ใน hash mode


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def simple_tokenize(text: str) -> List[str]:
    # ตัดคำด้วยช่องว่าง: "a b  c" -> ['a', 'b', 'c']  (state ทั้งหมดคิดระดับ word)
    return text.strip().split() if text and text.strip() else []


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (n, d), b: (m, d) -> (n, m) ค่า cos similarity ทุกคู่
    # normalize ให้ยาว 1 ก่อน แล้ว dot product = cos (ค่าอยู่ช่วง -1 ถึง 1)
    a = nn.functional.normalize(a, p=2, dim=-1)
    b = nn.functional.normalize(b, p=2, dim=-1)
    return a @ b.T


class TokenEncoder:
    """แปลงคำ (whitespace word) เป็น vector

    hf mode: subword vector ถูก mean-pool กลับเป็นระดับ word และ encode ทีละ chunk
    เมื่อ context ยาวเกิน limit ของ encoder
    """

    def __init__(self, config: Optional[EncoderConfig] = None, device=None):
        self.config = config or EncoderConfig()
        if self.config.mode not in ('hash', 'hf'):
            raise ValueError('encoder mode must be hash or hf')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        if self.config.mode == 'hf':
            from transformers import AutoModel, AutoTokenizer
            name = self.config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            if not getattr(self.tokenizer, 'is_fast', False):
                raise ValueError(f'{name}: hf mode needs a fast tokenizer (word_ids())')
            self.model = AutoModel.from_pretrained(name).to(self.device)
            self.model.eval()

    def _hash_vector(self, token: str) -> torch.Tensor:
        # hash mode: เอา md5 ของคำเป็น seed แล้วสุ่ม vector
        # คำเดียวกันได้ vector เดิมเสมอ (cos = 1) คำต่างกันได้ vector แทบตั้งฉาก (cos ~ 0)
        # ไม่มีความหมายทางภาษา ใช้ทดสอบว่าโค้ดทำงานเท่านั้น
        seed = int.from_bytes(hashlib.md5(token.lower().encode('utf-8')).digest()[:4], 'little')
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.config.hash_dim).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        return torch.tensor(v)

    def _max_chunk_len(self) -> int:
        # ความยาวสูงสุดที่ encoder รับได้ต่อครั้ง ลบ 2 ไว้ให้ [CLS] กับ [SEP]
        # MiniLM รับ 512 -> chunk ละ 510 subword
        limit = getattr(self.model.config, 'max_position_embeddings', 512) or 512
        tok_limit = getattr(self.tokenizer, 'model_max_length', limit)
        if not isinstance(tok_limit, int) or tok_limit > 100_000:
            tok_limit = limit
        return max(8, min(limit, tok_limit) - 2)

    def encode(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """text -> (list ของคำ, tensor ขนาด (จำนวนคำ, hidden))

        hf mode ทำงานแบบนี้
          1. ตัดเป็นคำ แล้วให้ tokenizer แตกเป็น subword ("running" -> "run", "##ning")
          2. ถ้ายาวเกิน limit ตัดเป็น chunk แล้ว encode ทีละ chunk
          3. word_ids() บอกว่า subword ไหนเป็นของคำไหน -> เฉลี่ย subword กลับเป็น 1 vector ต่อคำ
        """
        words = simple_tokenize(text)
        if self.config.mode == 'hash':
            if not words:
                return [], torch.empty((0, self.config.hash_dim))
            return words, torch.stack([self._hash_vector(t) for t in words])

        hidden = self.model.config.hidden_size
        if not words:
            return [], torch.empty((0, hidden))
        enc = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False, truncation=False)
        ids: List[int] = enc['input_ids']
        word_ids = enc.word_ids()
        max_len = self._max_chunk_len()
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        sub_vecs: List[torch.Tensor] = []
        for s in range(0, len(ids), max_len):
            chunk = ids[s:s + max_len]
            full = ([cls_id] if cls_id is not None else []) + chunk + ([sep_id] if sep_id is not None else [])
            x = torch.tensor([full], device=self.device)
            with torch.no_grad():
                h = self.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state.squeeze(0)
            if cls_id is not None:
                h = h[1:]
            sub_vecs.append(h[:len(chunk)].detach().cpu())
        sub = torch.cat(sub_vecs, dim=0) if sub_vecs else torch.empty((0, hidden))
        # รวม subword เข้าคำของมัน: vecs[คำ] += subword, counts[คำ] += 1 แล้วหารเฉลี่ย
        vecs = torch.zeros((len(words), hidden))
        counts = torch.zeros(len(words))
        for i, w in enumerate(word_ids):
            if w is not None:
                vecs[w] += sub[i]
                counts[w] += 1
        return words, vecs / counts.clamp(min=1).unsqueeze(1)


@dataclass
class StateFeatures:
    input_tokens: float     # [0] ความยาว context / 4096 (สูงสุด 2.0)  เช่น 2048 คำ -> 0.5
    relevance_mean: float   # [1] โดยรวม context เกี่ยวกับคำถามแค่ไหน
    relevance_max: float    # [2] มีคำที่เกี่ยวกับคำถามมากๆ อยู่ไหม
    mig_mean: float         # [3] ข้อมูลที่ "เกี่ยวและไม่ซ้ำ" โดยเฉลี่ย
    mig_max: float          # [4] คำที่คุ้มค่าที่สุด
    mig_std: float          # [5] ข้อมูลสำคัญกระจุกที่เดียว (std สูง) หรือกระจายทั่ว (std ต่ำ)
    redundancy_mean: float  # [6] เนื้อหาซ้ำกันแค่ไหน (สูง = บีบได้มาก)

    def to_numpy(self) -> np.ndarray:
        return np.asarray([getattr(self, n) for n in STATE_NAMES], dtype=np.float32)


class StateBuilder:
    """context + query -> State 7 ตัว

    relevance  = cos(word, mean(query))
    redundancy = cos สูงสุดกับคำอื่นใน context (ไม่นับตัวเอง)
                 ถ้า n > max_pairwise ใช้ anchor กระจายเท่าๆ กัน max_pairwise ตัว
    MIG        = relevance - redundancy
    """

    def __init__(self, encoder: TokenEncoder, max_pairwise: int = 512, normalize_length_by: float = 4096.0):
        if max_pairwise < 2 or normalize_length_by <= 0:
            raise ValueError('invalid max_pairwise or normalize_length_by')
        self.encoder = encoder
        self.max_pairwise = max_pairwise
        self.normalize_length_by = normalize_length_by

    def build(self, context: str, query: str) -> Tuple[StateFeatures, int]:
        """return (state 7 ตัว, จำนวนคำใน context)"""
        # 1) encode ทั้ง context และ query เป็น vector ระดับคำ
        ctoks, cvecs = self.encoder.encode(context)
        qtoks, qvecs = self.encoder.encode(query)
        n = len(ctoks)
        if n == 0:   # context ว่าง -> state เป็น 0 ทั้งหมด
            return StateFeatures(0, 0, 0, 0, 0, 0, 0), 0

        # 2) query เหลือ 1 vector = ค่าเฉลี่ยของทุกคำในคำถาม
        q = qvecs.mean(dim=0, keepdim=True) if qtoks else torch.zeros((1, cvecs.shape[-1]))

        # 3) relevance[i] = cos(คำที่ i, query)  -> shape (n,)
        relevance = cosine_matrix(cvecs, q).squeeze(1)

        # 4) redundancy[i] = คำที่ i ซ้ำกับคำอื่นมากที่สุดแค่ไหน
        if n == 1:
            redundancy = torch.zeros_like(relevance)   # มีคำเดียว ไม่มีอะไรให้ซ้ำ
        elif n <= self.max_pairwise:
            # context สั้น: เทียบทุกคู่ (n x n) แล้วตั้งเส้นทแยง = -1 เพื่อไม่นับการเทียบกับตัวเอง
            sim = cosine_matrix(cvecs, cvecs)
            sim.fill_diagonal_(-1.0)
            redundancy = sim.max(dim=1).values
        else:
            # context ยาว: เทียบทุกคำกับ anchor 512 คำที่เลือกห่างเท่าๆ กัน (ประหยัด memory)
            # เช่น n=2048 -> anchor คือคำที่ 0, 4, 8, ..., 2047
            idx = torch.linspace(0, n - 1, self.max_pairwise).long()
            sim = cosine_matrix(cvecs, cvecs[idx])
            sim[idx, torch.arange(len(idx))] = -1.0    # anchor ไม่นับเทียบกับตัวเอง
            redundancy = sim.max(dim=1).values

        # 5) MIG = เกี่ยวกับคำถาม - ซ้ำกับคำอื่น  (สูง = คำนี้มีประโยชน์และไม่ซ้ำ)
        mig = relevance - redundancy

        # 6) ย่อ vector ยาว n ให้เหลือ 7 ตัวเลข (mean / max / std) เพื่อเป็น input ของ DQN
        state = StateFeatures(
            input_tokens=min(n / self.normalize_length_by, 2.0),
            relevance_mean=float(relevance.mean()), relevance_max=float(relevance.max()),
            mig_mean=float(mig.mean()), mig_max=float(mig.max()),
            mig_std=float(mig.std(unbiased=False)), redundancy_mean=float(redundancy.mean()),
        )
        if not np.isfinite(state.to_numpy()).all():
            raise ValueError('non-finite state')
        return state, n


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
class RewardCalculator:
    """แปลงผลจาก COMI เป็น reward

    ตัวอย่าง (cost_source='tokens')
        2048 -> 128 token, quality 0.80 : 0.55*0.80 + 0.45*0.9375        =  0.862
        2048 ->  64 token, quality 0.65 : 0.55*0.65 + 0.45*0.96875 - 1.0 = -0.207  (โดน penalty)
    """

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
            # ไม่มีค่าวัดจริง -> ใช้ token_saving แทน energy และ latency
            # weight รวมของ token = 0.20 + 0.15 + 0.10 = 0.45
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
        # สัดส่วนที่ลดลง: 1 - หลังบีบ/ก่อนบีบ  เช่น 1 - 128/2048 = 0.9375
        # ค่าลบ = หลังบีบยาวกว่าเดิม (ไม่ควรเกิด) ตัดไว้ที่ -1
        if original_tokens <= 0:
            raise ValueError('original_tokens must be > 0')
        if compressed_tokens < 0:
            raise ValueError('compressed_tokens must be >= 0')
        return float(np.clip(1.0 - compressed_tokens / original_tokens, -1, 1))


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    # ประสบการณ์ 1 ครั้ง: เห็น state -> เลือก action -> ได้ reward
    state: np.ndarray       # state 7 ตัวตอนตัดสินใจ
    action: int             # action index ที่เลือก
    reward: float           # reward ที่ได้จาก COMI
    next_state: np.ndarray  # งานนี้ไม่มี state ถัดไป ใส่ 0 ไว้
    done: bool              # True เสมอ (1 request = 1 decision จบ)


class ReplayBuffer:
    # เก็บประสบการณ์ไว้แล้วสุ่มมา train ทีละ batch
    # ช่วยให้ไม่ train ติดกันแต่ข้อมูลที่เพิ่งเห็น (ลด correlation ระหว่าง sample)
    def __init__(self, capacity: int = 5000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    # MLP: state 7 ตัว -> 64 -> ReLU -> 64 -> ReLU -> Q ของแต่ละ action
    # Q[a] = reward ที่คาดว่าจะได้ถ้าเลือก action a ใน state นี้
    # เช่น merge_sizes 16,32 -> output 2 ค่า [Q(16), Q(32)] แล้วเลือกตัวที่มากกว่า
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
    """DQN ทั่วไป ไม่รู้ว่า state/action หมายถึงอะไร (ความหมายอยู่ที่ PromptController)

    policy_net : ตัวที่ใช้เลือก action และถูก train
    target_net : สำเนาของ policy ไว้คิด target (copy ทุก target_update_every update)
    """

    def __init__(self, config: DQNConfig, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        hidden = (config.hidden_1, config.hidden_2)
        self.policy_net = QNetwork(config.state_dim, config.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.total_steps = 0   # จำนวน gradient update
        self.env_steps = 0     # จำนวนครั้งที่เลือก action ตอน train (ใช้ลด epsilon)

    def epsilon(self) -> float:
        # โอกาสสุ่ม action ลดแบบเส้นตรง
        # eps = 1.0 + min(1, step/decay) * (0.05 - 1.0)
        # decay=1000: step 0 -> 1.0, step 500 -> 0.525, step >= 1000 -> 0.05
        c = self.config
        f = min(1.0, self.env_steps / max(c.epsilon_decay_steps, 1))
        return c.epsilon_start + f * (c.epsilon_end - c.epsilon_start)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(x).squeeze(0).cpu().numpy()

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, bool]:
        """return (action, explored). evaluation = greedy และไม่นับ env_steps"""
        if training:
            eps = self.epsilon()
            self.env_steps += 1
            if random.random() < eps:
                return random.randrange(self.config.action_dim), True
        return int(np.argmax(self.q_values(state))), False

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(Transition(np.asarray(state, dtype=np.float32), int(action), float(reward),
                                    np.asarray(next_state, dtype=np.float32), bool(done)))

    def train_step(self) -> Optional[float]:
        """update policy_net 1 ครั้ง คืนค่า loss (หรือ None ถ้า replay ยังไม่พอ)

        target = r + gamma * max Q_target(s') * (1 - done)
        งานนี้ gamma = 0 และ done = True -> target = r
        แปลว่า DQN เรียนให้ Q(s, a) เข้าใกล้ reward จริงที่ได้จากการเลือก a ใน s
        """
        c = self.config
        if len(self.replay) < max(c.batch_size, c.warmup_steps):
            return None
        b = self.replay.sample(c.batch_size)
        dev = self.device
        states = torch.tensor(np.stack([x.state for x in b]), dtype=torch.float32, device=dev)
        actions = torch.tensor([x.action for x in b], dtype=torch.long, device=dev).unsqueeze(1)
        rewards = torch.tensor([x.reward for x in b], dtype=torch.float32, device=dev).unsqueeze(1)
        next_states = torch.tensor(np.stack([x.next_state for x in b]), dtype=torch.float32, device=dev)
        dones = torch.tensor([x.done for x in b], dtype=torch.float32, device=dev).unsqueeze(1)
        # Q ของ action ที่เลือกไปจริง (gather หยิบเฉพาะคอลัมน์ของ action นั้น)
        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + c.gamma * next_q * (1.0 - dones)
        # Huber loss: error เล็กใช้แบบ MSE error ใหญ่ใช้แบบ L1 (ไม่ระเบิดตอน reward โดน penalty -1)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)   # กัน gradient ใหญ่เกิน
        self.optimizer.step()
        self.total_steps += 1
        if self.total_steps % c.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())


# ---------------------------------------------------------------------------
# Controller (ตัวที่ COMI เรียกใช้)
# ---------------------------------------------------------------------------
@dataclass
class ControllerDecision:
    """ผลการตัดสินใจ 1 request ส่งต่อให้ COMI

    COMI ใช้ compress_rate (หรือ merge_size ถ้า action มาจาก --merge-sizes)
    แล้วส่ง quality + compressed_tokens กลับมาพร้อม decision นี้
    """
    request_id: str            # id ของ request ไว้จับคู่กับผลจาก COMI
    action_index: int          # index ใน actions ที่ DQN เลือก
    action_name: str           # เช่น 'MERGE_16'
    retention_ratio: float     # สัดส่วนที่เก็บ เช่น 1/16 = 0.0625
    compress_rate: float       # 1 / retention เช่น 16.0
    input_tokens: int          # จำนวนคำใน context (นับแบบ whitespace)
    target_tokens: int         # ประมาณจำนวนที่เหลือ = ceil(input_tokens * retention) เช่น 2048 -> 128
    state: List[float]         # state 7 ตัว เก็บไว้ใช้ตอน train
    q_values: List[float]      # Q ของทุก action ตอนตัดสินใจ (ไว้ดูว่ามั่นใจแค่ไหน)
    explored: bool             # True = action นี้มาจากการสุ่ม (epsilon) ไม่ใช่จาก Q
    epsilon: Optional[float]   # epsilon ตอนนั้น (None ถ้าเป็นโหมด greedy)
    feature_version: str = FEATURE_VERSION
    merge_size: Optional[int] = None   # compress_rate ถ้าเป็นจำนวนเต็ม (ใช้กับ COMI ได้ตรงๆ) ไม่งั้น None

    def to_dict(self) -> Dict:
        return asdict(self)


class PromptController:
    """ตัวที่ code อื่นเรียกใช้ รวม encoder + state + DQN + reward ไว้ที่เดียว

    วิธีใช้ 3 ขั้น
        d = ctl.decide(context, query)                       # 1. ได้ action / merge_size
        ... COMI บีบตาม d.merge_size แล้ว generate ...         # 2. ทำนอก controller
        ctl.feedback(d, quality=f1, compressed_tokens=slots)  # 3. ส่งผลกลับมา train
    """

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

    # ---- decide ----------------------------------------------------------
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
        action, explored = self.agent.select_action(state, training=explore)
        spec = self.actions[action]
        # merge_size ใส่ให้เฉพาะเมื่อ compress_rate เป็นจำนวนเต็ม (เช่น 16.0 -> 16, แต่ 1.333 -> None)
        merge = round(spec.compress_rate)
        return ControllerDecision(
            request_id=str(request_id) if request_id is not None else '',
            action_index=action, action_name=spec.name,
            retention_ratio=spec.retention, compress_rate=spec.compress_rate,
            input_tokens=int(input_tokens),
            target_tokens=max(1, math.ceil(input_tokens * spec.retention)) if input_tokens else 0,
            state=state.tolist(), q_values=self.agent.q_values(state).tolist(),
            explored=explored, epsilon=eps,
            merge_size=merge if abs(merge - spec.compress_rate) < 1e-6 else None,
        )

    # ---- feedback จาก COMI ----------------------------------------------
    def _check_decision(self, d: Dict):
        # กันเอา feedback ของ controller ตัวอื่น (action ชุดอื่น / state สูตรอื่น) มา train ปนกัน
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
        """รับผลจาก COMI -> reward -> replay -> (train_step)

        original_tokens: จำนวน token ก่อนบีบที่ COMI นับเอง ถ้าไม่ส่งมาใช้ input_tokens (word) ของ decision

        ค่าที่ต้องส่งมา
            decision          : ตัวที่ได้จาก decide() (หรือ dict จากไฟล์ decisions.jsonl)
            quality           : F1 ของคำตอบเทียบ reference ช่วง [0, 1]
            compressed_tokens : จำนวน memory slot ที่ COMI สร้างจริง
            train             : True = update DQN ทันที 1 ครั้ง, False = เก็บเข้า replay อย่างเดียว
        """
        d = decision.to_dict() if isinstance(decision, ControllerDecision) else dict(decision)
        self._check_decision(d)
        original = int(original_tokens if original_tokens is not None else d['input_tokens'])
        saving = self.reward_calc.token_saving(original, int(compressed_tokens))
        reward = self.reward_calc.calculate(quality, saving, energy_saving, latency_saving)
        state = np.asarray(d['state'], dtype=np.float32)
        # 1 decision ต่อ request -> done=True, next_state ไม่ถูกใช้ (gamma=0)
        self.agent.remember(state, int(d['action_index']), reward, np.zeros_like(state), True)
        loss = self.agent.train_step() if train else None
        return {'reward': reward, 'token_saving': saving, 'loss': loss}

    def train(self, updates: int) -> List[float]:
        # update หลายครั้งรวด ใช้ตอน train จากไฟล์ feedback ทีเดียว
        losses = []
        for _ in range(updates):
            loss = self.agent.train_step()
            if loss is not None:
                losses.append(loss)
        return losses

    # ---- checkpoint ------------------------------------------------------
    def save(self, path, include_replay: bool = True):
        # เก็บทุกอย่างที่ต้องใช้สร้าง controller ตัวเดิมกลับมา:
        # action table, encoder config, weight, optimizer, step counter (epsilon) และ replay
        a = self.agent
        payload = {
            'feature_version': FEATURE_VERSION,
            'actions': [[s.name, s.retention] for s in self.actions],
            'encoder': asdict(self.encoder.config),
            'max_pairwise': self.state_builder.max_pairwise,
            'normalize_length_by': self.state_builder.normalize_length_by,
            'dqn_config': asdict(a.config),
            'reward_config': asdict(self.reward_calc.config),
            'policy': a.policy_net.state_dict(), 'target': a.target_net.state_dict(),
            'optimizer': a.optimizer.state_dict(),
            'total_steps': a.total_steps, 'env_steps': a.env_steps,
        }
        if include_replay and len(a.replay):
            buf = list(a.replay.buffer)
            payload['replay'] = {
                'state': torch.tensor(np.stack([t.state for t in buf])),
                'action': torch.tensor([t.action for t in buf]),
                'reward': torch.tensor([t.reward for t in buf], dtype=torch.float32),
                'next_state': torch.tensor(np.stack([t.next_state for t in buf])),
                'done': torch.tensor([t.done for t in buf]),
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, encoder: Optional[TokenEncoder] = None, device=None) -> 'PromptController':
        # weights_only=True: โหลดได้เฉพาะ tensor/ค่าพื้นฐาน ปลอดภัยกว่า pickle ทั่วไป
        p = torch.load(path, map_location=device or 'cpu', weights_only=True)
        if p.get('feature_version') != FEATURE_VERSION:
            raise ValueError('checkpoint uses a different feature definition')
        enc_cfg = EncoderConfig(**p['encoder'])
        if encoder is None:
            encoder = TokenEncoder(enc_cfg, device=device)
        elif asdict(encoder.config) != asdict(enc_cfg):
            raise ValueError('encoder config does not match checkpoint')
        ctl = cls(encoder, [ActionSpec(n, float(r)) for n, r in p['actions']],
                  DQNConfig(**p['dqn_config']), RewardConfig(**p['reward_config']),
                  p['max_pairwise'], p['normalize_length_by'], device=device)
        a = ctl.agent
        a.policy_net.load_state_dict(p['policy'])
        a.target_net.load_state_dict(p['target'])
        a.optimizer.load_state_dict(p['optimizer'])
        a.total_steps, a.env_steps = int(p['total_steps']), int(p['env_steps'])
        r = p.get('replay')
        if r is not None:
            for i in range(len(r['action'])):
                a.remember(r['state'][i].numpy(), int(r['action'][i]), float(r['reward'][i]),
                           r['next_state'][i].numpy(), bool(r['done'][i]))
        return ctl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ใช้เมื่อต้องการรัน controller กับ COMI แยกขั้นกันผ่านไฟล์
#   init   : สร้าง checkpoint ใหม่
#   decide : requests.jsonl  -> decisions.jsonl (ให้ COMI อ่าน)
#   train  : feedback.jsonl  -> checkpoint ที่ train แล้ว
# ถ้ารันพร้อม COMI ใน process เดียว ใช้ run_dqn_over_comi.py แทน

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
        ctl.save(args.checkpoint)   # เก็บ env_steps เพื่อให้ epsilon ลดต่อเนื่อง
    counts: Dict[str, int] = {}
    for d in out:
        counts[d['action_name']] = counts.get(d['action_name'], 0) + 1
    print(json.dumps({'decisions': len(out), 'action_counts': counts, 'output': args.output}, ensure_ascii=False))


def cmd_train(args):
    # 1) โหลด feedback ทุกแถวเข้า replay (train=False)  2) update DQN ตามจำนวน --updates  3) save
    ctl = PromptController.load(args.checkpoint)
    rewards = []
    for path in args.feedback:
        for row in read_jsonl(path):
            fb = ctl.feedback(row, float(row['quality']), int(row['compressed_tokens']),
                              row.get('original_tokens'), row.get('latency_saving'),
                              row.get('energy_saving'), train=False)
            rewards.append(fb['reward'])
    updates = args.updates if args.updates is not None else len(rewards)
    losses = ctl.train(updates)
    ctl.save(args.output or args.checkpoint)
    print(json.dumps({'feedback_rows': len(rewards),
                      'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
                      'updates_done': len(losses),
                      'loss_mean': float(np.mean(losses)) if losses else None,
                      'replay_size': len(ctl.agent.replay),
                      'epsilon': ctl.agent.epsilon()}, ensure_ascii=False))
    if not losses:
        print(f'[WARN] replay {len(ctl.agent.replay)} < warmup '
              f'{max(ctl.agent.config.batch_size, ctl.agent.config.warmup_steps)}: no update', file=sys.stderr)


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
    s.add_argument('--updates', type=int, help='gradient updates (default = feedback rows)')
    s.add_argument('--output', help='save to another path (default overwrite --checkpoint)')
    s.set_defaults(func=cmd_train)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    args.func(args)


if __name__ == '__main__':
    main()
