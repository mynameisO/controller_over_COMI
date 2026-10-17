"""DQN ครอบ COMI

DQN (prompt_controller.py) เป็น standalone ไม่ import และไม่แก้โค้ดของ COMI
ไฟล์นี้เป็นตัวกลางที่ถือทั้ง 2 ตัวไว้:

    sample (context, query, answer)
      -> PromptController.decide()      เลือก merge_size
      -> ตั้ง comi.merge_size ชั่วคราว แล้วเรียก comi(...) ตามปกติ
      -> นับ memory slot ที่ COMI สร้างจริง + วัด F1 ของคำตอบ
      -> PromptController.feedback()    reward -> replay -> train_step
      -> คืนค่า comi.merge_size / is_random เดิม

COMI ถูกเรียกผ่าน interface ปกติเท่านั้น: tokenizer, merge_size, is_random,
build_comi_memory และ forward(..., labels=None) ซึ่งคือ generate
"""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from prompt_controller import PromptController


# ---------------------------------------------------------------------------
# quality = SQuAD F1 (สูตรเดียวกับ eval_utils.py ของ COMI)
# ---------------------------------------------------------------------------
# quality_fn ของ wrapper คือ squad_f1 (เปลี่ยนเป็น metric อื่นได้ตอนสร้าง DQNOverCOMI)
# ตัวอย่าง: คำตอบ "The Paris." กับ reference "paris"
#   normalize -> "paris" ทั้งคู่ -> F1 = 1.0
# ถ้า reference เป็น list (มีหลายคำตอบที่ถูก) ใช้ค่าที่สูงสุด

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


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class DQNOverCOMI:
    """controller เลือก merge_size ต่อ sample แล้วให้ COMI บีบ + generate

    comi       : COMI model ที่โหลด checkpoint แล้ว (เรียกแค่ผ่าน interface ปกติ ไม่แก้ class)
    controller : PromptController ที่ action สร้างจาก merge_sizes (เช่น '16,32')
    quality_fn : (คำตอบ, reference) -> คะแนน [0, 1]  default = SQuAD F1
    """

    def __init__(self, comi, controller: PromptController,
                 quality_fn: Callable[[str, object], float] = squad_f1, device=None):
        # ทุก action ต้องแปลงเป็น merge_size จำนวนเต็มได้ เพราะ COMI รับ merge_size เป็น int
        missing = [a.name for a in controller.actions if round(a.compress_rate) != a.compress_rate]
        if missing:
            raise ValueError(f'actions {missing} have no integer merge_size; '
                             'create the controller with --merge-sizes')
        self.comi = comi
        self.controller = controller
        self.quality_fn = quality_fn
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

    def _row(self, sample, response, original, compressed, decision=None, merge_size=None):
        # รวมผลของ 1 sample: F1, EM, token ก่อน/หลังบีบ, saving, reward (สูตรเดียวกับ controller)
        quality = float(np.clip(self.quality_fn(response, sample.get('answer', '')), 0, 1))
        rc = self.controller.reward_calc
        saving = rc.token_saving(max(original, 1), compressed)
        row = {
            'sample_id': sample.get('sample_id'),
            'merge_size': merge_size if decision is None else decision.merge_size,
            'model_output': response, 'ground_truth': sample.get('answer', ''),
            'f1': quality, 'em': exact_match(response, sample.get('answer', '')),
            'original_tokens': original, 'compressed_tokens': compressed,
            'token_saving': saving, 'reward': rc.calculate(quality, saving),
        }
        if decision is not None:
            row.update(action_name=decision.action_name, explored=decision.explored,
                       state=decision.state, q_values=decision.q_values)
        return row

    def run(self, samples: Sequence[Dict], explore: bool = False, train: bool = False,
            batch_size: int = 1) -> List[Dict]:
        """explore/train=True ตอน train DQN, False ทั้งคู่ตอน evaluate (greedy)

        samples: list ของ {"input": context, "prompt": คำถาม, "answer": คำตอบ, "sample_id": ...}
                 (รูปแบบเดียวกับไฟล์ test ของ ft_inference.py)

        ต่อ 1 batch
          1. DQN เลือก merge_size ให้ทุก sample
          2. จัดกลุ่ม sample ที่ได้ merge_size เดียวกัน
             เช่น batch 4 อัน ได้ [16, 32, 16, 32] -> กลุ่ม 16 = [0, 2], กลุ่ม 32 = [1, 3]
          3. ส่งแต่ละกลุ่มเข้า COMI -> ได้คำตอบ + จำนวน token
          4. ถ้า train=True ส่ง F1 + compressed_tokens กลับให้ controller.feedback()
          5. เรียงผลกลับตามลำดับเดิมของ sample
        """
        rows: List[Dict] = []
        for start in range(0, len(samples), batch_size):
            batch = list(samples[start:start + batch_size])
            # ขั้น 1: DQN ดู context + query ของแต่ละ sample แล้วเลือก action
            decisions =[self.controller.decide(s.get('input', ''), s.get('prompt', ''),
                                                s.get('sample_id'), explore=explore) for s in batch]
            groups = defaultdict(list)      # batch ตามค่า merge_size เพราะ COMI ใช้ค่าเดียวทั้ง batch
            for i, d in enumerate(decisions):
                groups[d.merge_size].append(i)
            out: Dict[int, Dict] = {}
            for merge_size, idxs in groups.items():
                responses, original, compressed = self._generate([batch[i] for i in idxs], merge_size)
                for k, i in enumerate(idxs):
                    row = self._row(batch[i], responses[k], original[k], compressed[k], decisions[i])
                    if train:
                        # ขั้น 4: quality = F1, compressed = memory slot จริง, original = token ก่อนบีบ
                        fb = self.controller.feedback(decisions[i], row['f1'], compressed[k], original[k])
                        row['loss'] = fb['loss']
                    out[i] = row
            rows.extend(out[i] for i in range(len(batch)))
        return rows

    def run_fixed(self, samples: Sequence[Dict], merge_size: int, batch_size: int = 1) -> List[Dict]:
        """baseline: COMI ใช้ merge_size คงที่ ไม่ผ่าน DQN

        ใช้เทียบใน eval เช่น Fixed-16 vs Fixed-32 vs DQN ถ้า DQN ไม่ชนะ Fixed ทุกตัว
        แปลว่ายังไม่คุ้มที่จะใช้ controller
        """
        rows = []
        for start in range(0, len(samples), batch_size):
            batch = list(samples[start:start + batch_size])
            responses, original, compressed = self._generate(batch, merge_size)
            rows.extend(self._row(s, r, o, c, merge_size=merge_size)
                        for s, r, o, c in zip(batch, responses, original, compressed))
        return rows


def summarize(rows: Sequence[Dict]) -> Dict:
    # ค่าเฉลี่ยของ reward / F1 / EM / saving + นับว่าแต่ละ merge_size ถูกเลือกกี่ครั้ง
    if not rows:
        return {'n': 0}
    counts: Dict[str, int] = {}
    for r in rows:
        key = str(r.get('merge_size'))
        counts[key] = counts.get(key, 0) + 1
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {'n': len(rows), 'reward': mean('reward'), 'f1': mean('f1'), 'em': mean('em'),
            'token_saving': mean('token_saving'),
            'compressed_tokens': mean('compressed_tokens'), 'merge_size_counts': counts}
