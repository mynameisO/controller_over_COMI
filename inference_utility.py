"""Text encoding, state features, COMI restoration, generation, and QA metrics.

COMI/PEFT/Transformers imports are lazy so hash-encoder controller workflows
and CPU tests do not require the LLM dependencies or model downloads.
"""
from __future__ import annotations

import hashlib
import random
import re
import string
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


FEATURE_VERSION = 'prompt-word-r-minus-d-v1'

STATE_NAMES = ('input_tokens', 'relevance_mean', 'relevance_max',
               'mig_mean', 'mig_max', 'mig_std', 'redundancy_mean')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class EncoderConfig:
    mode: str = 'hash'
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    hash_dim: int = 128

def simple_tokenize(text: str) -> List[str]:
    return text.split() if text else []

def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(a, p=2, dim=-1)
    b = nn.functional.normalize(b, p=2, dim=-1)
    return a @ b.T

class TokenEncoder:
    """Encode whitespace words with test hashes or a chunked HF encoder."""

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
        seed = int.from_bytes(hashlib.md5(token.lower().encode('utf-8')).digest()[:4], 'little')
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.config.hash_dim).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        return torch.tensor(v)

    def _max_chunk_len(self) -> int:
        limit = getattr(self.model.config, 'max_position_embeddings', 512) or 512
        tok_limit = getattr(self.tokenizer, 'model_max_length', limit)
        if not isinstance(tok_limit, int) or tok_limit > 100_000:
            tok_limit = limit
        return max(8, min(limit, tok_limit) - 2)

    def encode(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """Return words and vectors; mean-pool HF subwords back to words."""
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

        vecs = torch.zeros((len(words), hidden))
        counts = torch.zeros(len(words))
        for i, w in enumerate(word_ids):
            if w is not None:
                vecs[w] += sub[i]
                counts[w] += 1
        return words, vecs / counts.clamp(min=1).unsqueeze(1)

@dataclass
class StateFeatures:
    input_tokens: float
    relevance_mean: float
    relevance_max: float
    mig_mean: float
    mig_max: float
    mig_std: float
    redundancy_mean: float

    def to_numpy(self) -> np.ndarray:
        return np.asarray([getattr(self, n) for n in STATE_NAMES], dtype=np.float32)

class StateBuilder:
    """Build seven features: length, relevance, MIG, and redundancy.

    Relevance is similarity to the mean query vector; redundancy is maximum
    similarity to another context word, using anchors for long contexts.
    MIG = relevance - redundancy.
    """

    def __init__(self, encoder: TokenEncoder, max_pairwise: int = 512, normalize_length_by: float = 4096.0):
        if max_pairwise < 2 or normalize_length_by <= 0:
            raise ValueError('invalid max_pairwise or normalize_length_by')
        self.encoder = encoder
        self.max_pairwise = max_pairwise
        self.normalize_length_by = normalize_length_by

    def build(self, context: str, query: str) -> Tuple[StateFeatures, int]:
        """return (state 7 ตัว, จำนวนคำใน context)"""

        ctoks, cvecs = self.encoder.encode(context)
        qtoks, qvecs = self.encoder.encode(query)
        n = len(ctoks)
        if n == 0:
            return StateFeatures(0, 0, 0, 0, 0, 0, 0), 0

        q = qvecs.mean(dim=0, keepdim=True) if qtoks else torch.zeros((1, cvecs.shape[-1]))

        relevance = cosine_matrix(cvecs, q).squeeze(1)

        if n == 1:
            redundancy = torch.zeros_like(relevance)
        elif n <= self.max_pairwise:
            sim = cosine_matrix(cvecs, cvecs)
            sim.fill_diagonal_(-1.0)
            redundancy = sim.max(dim=1).values
        else:
            idx = torch.linspace(0, n - 1, self.max_pairwise).long()
            sim = cosine_matrix(cvecs, cvecs[idx])
            sim[idx, torch.arange(len(idx))] = -1.0
            redundancy = sim.max(dim=1).values

        mig = relevance - redundancy

        state = StateFeatures(
            input_tokens=min(n / self.normalize_length_by, 2.0),
            relevance_mean=float(relevance.mean()), relevance_max=float(relevance.max()),
            mig_mean=float(mig.mean()), mig_max=float(mig.max()),
            mig_std=float(mig.std(unbiased=False)), redundancy_mean=float(redundancy.mean()),
        )
        if not np.isfinite(state.to_numpy()).all():
            raise ValueError('non-finite state')
        return state, n


def normalize_answer(s: str) -> str:
    # ตัวพิมพ์เล็ก, ตัดเครื่องหมายวรรคตอน, ตัด a/an/the, ยุบช่องว่าง
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1_score(prediction: str, ground_truth: str) -> float:
    # F1 ระดับคำ: precision = คำที่ตรง/คำที่ตอบ, recall = คำที่ตรง/คำใน reference
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)

def squad_f1(prediction: str, answer) -> float:
    answers = answer if isinstance(answer, (list, tuple)) else [answer]
    return max(f1_score(prediction, str(a)) for a in answers) if answers else 0.0

def exact_match(prediction: str, answer) -> float:
    answers = answer if isinstance(answer, (list, tuple)) else [answer]
    return float(any(normalize_answer(prediction) == normalize_answer(str(a)) for a in answers))


def restore_comi_model(model_args, training_args, device, *, freeze: bool = True):
    """Build COMI, restore its optional checkpoint, and prepare for inference.

    Uses the same LoRA setup and non-strict checkpoint loading as the existing
    train/inference entry points. Only model weights are frozen; controller
    training remains independent. Imports occur only when called.
    """
    from peft import LoraConfig
    from modeling_comi import COMI, build_lora_config, load_checkpoint_file

    model_args.train = False
    model = COMI(model_args, training_args,
                 lora_config=LoraConfig(**build_lora_config(model_args)))
    if training_args.restore_from:
        state_dict, checkpoint = load_checkpoint_file(training_args.restore_from)
        print(f'Loading COMI checkpoint {checkpoint}')
        model.load_state_dict(state_dict, strict=False)
    if freeze:
        model.requires_grad_(False)
    model.to(device).eval()
    return model


class COMIInference:
    """Tokenize and generate with COMI while recording compression counts."""

    def __init__(self, comi, device=None):
        self.comi = comi
        self.device = device or next(comi.parameters()).device

    @contextmanager
    def _comi_merge_size(self, merge_size: int, record: Dict):
        """ตั้ง merge_size ชั่วคราว + จด memory mask ที่ COMI สร้าง แล้วคืนค่าเดิม

        ทำไมต้องจด mask: forward ของ COMI คืนแค่ token ที่ generate ออกมา ไม่บอกว่าบีบเหลือกี่ slot
        เลยครอบ build_comi_memory ของ instance นี้ไว้ชั่วคราว ให้เก็บ mask ลง record['memory_mask']
        mask[i].sum() = จำนวน memory slot ของ sample i = compressed_tokens ที่ใช้คิด reward

        ใช้กับ with:
            with self._comi_merge_size(16, record):
                comi(...)          # ข้างในนี้ COMI ใช้ merge_size=16, is_random=False
            # ออกจาก with แล้ว merge_size / is_random กลับเป็นค่าเดิม และ hook ถูกถอดออก
        """
        comi = self.comi
        old_merge, old_random = comi.merge_size, comi.is_random
        original_build = comi.build_comi_memory

        def recording_build(*args, **kwargs):
            memory, mask = original_build(*args, **kwargs)
            record['memory_mask'] = mask.detach()
            return memory, mask

        # is_random=False ไม่งั้น COMI จะสุ่ม merge_size ทับค่าที่ DQN เลือก
        comi.merge_size, comi.is_random = int(merge_size), False
        comi.build_comi_memory = recording_build      # instance attribute, ไม่แตะ class ของ COMI
        try:
            yield
        finally:
            del comi.build_comi_memory
            comi.merge_size, comi.is_random = old_merge, old_random

    def _generate(self, samples: Sequence[Dict], merge_size: int):
        """ส่ง sample ทั้งกลุ่มเข้า COMI ด้วย merge_size เดียวกัน

        tokenize แบบเดียวกับ ft_inference.py (padding=True, add_special_tokens=False)
        return
            responses  : คำตอบที่ COMI generate
            original   : จำนวน token ของ context ก่อนบีบ (นับจาก attention_mask ไม่นับ padding)
            compressed : จำนวน memory slot หลังบีบ (นับจาก memory mask ที่จดไว้)
        เช่น context 2048 token, merge_size 16 -> original 2048, compressed ~128
        """
        tok = self.comi.tokenizer
        ctx = tok([s.get('input', '') for s in samples], return_tensors='pt', padding=True, add_special_tokens=False)
        qry = tok([s.get('prompt', '') for s in samples], return_tensors='pt', padding=True, add_special_tokens=False)
        record: Dict = {}
        with torch.inference_mode(), self._comi_merge_size(merge_size, record):
            out = self.comi(
                input_ids=ctx['input_ids'].to(self.device),
                attention_mask=ctx['attention_mask'].to(self.device),
                query_ids=qry['input_ids'].to(self.device),
                query_attention_mask=qry['attention_mask'].to(self.device),
                labels=None,
            )
        responses = tok.batch_decode(out, skip_special_tokens=True)
        original = ctx['attention_mask'].sum(dim=1).tolist()
        compressed = record['memory_mask'].sum(dim=1).tolist()
        return responses, [int(x) for x in original], [int(x) for x in compressed]
