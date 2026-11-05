"""Connect the compression controller to COMI generation and feedback.

Encoding, model restoration, generation, and QA metrics live in
inference_utility; policy learning and checkpoints live in dqn_network.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import numpy as np

from dqn_network import PromptController
from inference_utility import (
    COMIInference, exact_match, f1_score, normalize_answer, squad_f1,
)


class DQNOverCOMI(COMIInference):
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
        super().__init__(comi, device=device)
        self.controller = controller
        self.quality_fn = quality_fn

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
