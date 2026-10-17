# DQN Controller ครอบ COMI

DQN เป็น standalone (`prompt_controller.py` ไม่ import โค้ด COMI) และครอบ COMI จากด้านนอก
โดยไม่แก้ไฟล์ใดๆ ของ COMI

```
sample ──> DQN (standalone) ──merge_size──> COMI (บีบ + generate) ──> คำตอบ
             ▲                                                       │
             └──── reward <── F1, memory slot ที่ COMI สร้างจริง ─────┘
```

`comi_dqn.py` เป็นตัวกลางระหว่างสองฝั่ง ต่อ 1 sample จะทำตามลำดับนี้
1. controller ดู context + query แล้วเลือก `merge_size`
2. ตั้ง `comi.merge_size` ชั่วคราว (ปิด `is_random`) แล้วเรียก `comi(..., labels=None)` ตามปกติ
3. นับ memory slot จาก mask ที่ `build_comi_memory` คืนออกมา และนับ token ก่อนบีบจาก attention mask
4. วัด F1 ของคำตอบ (สูตรเดียวกับ `eval_utils.py`) → reward → replay → `train_step`
5. คืนค่า `merge_size` / `is_random` เดิมให้ COMI

ถ้าใน batch มีหลาย sample จะแยกกลุ่มตาม merge_size ก่อนส่งเข้า COMI เพราะ COMI ใช้ merge_size ค่าเดียวทั้ง batch
ตัว COMI ไม่ถูก train (เป็น `eval()` + `inference_mode`) ที่ถูก train มีแค่ DQN

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `prompt_controller.py` | DQN standalone: encoder, State, DQNAgent, Reward, CLI แบบไฟล์ |
| `comi_dqn.py` | `DQNOverCOMI` ครอบ COMI (duck-typed ไม่ import COMI) |
| `run_dqn_over_comi.py` | script train / eval ที่โหลด COMI จริง |
| `tests/test_controller.py` | test ของ controller |
| `tests/test_comi_wrapper.py` | test ของ wrapper กับ COMI ปลอม |

## Train / Eval กับ COMI

วางโฟลเดอร์นี้ไว้ใน repo COMI (ค่า default ของ `--dqn_comi_dir` คือโฟลเดอร์แม่) argument ของ COMI ใช้ชื่อเดิมทั้งหมด
ส่วนของ DQN ขึ้นต้นด้วย `--dqn_` ข้อมูลใช้รูปแบบเดียวกับ `ft_inference.py`: `{"input", "prompt", "answer"}`

```bash
# train DQN (COMI ไม่ถูก train)
python run_dqn_over_comi.py --dqn_mode train \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --restore_from /path/to/comi_ckpt \
  --merge_sizes 4,8,16,32 --segment_size 10000 \
  --train_file data/train.jsonl --output_dir runs/dqn --dqn_epochs 5 --report_to none

# eval: DQN (greedy) เทียบกับ fixed merge_size ทุกค่า
python run_dqn_over_comi.py --dqn_mode eval --dqn_checkpoint runs/dqn/dqn_controller.pt \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --restore_from /path/to/comi_ckpt \
  --test_file data/test.jsonl --output_dir runs/dqn --per_device_eval_batch_size 4 --report_to none
```

- action ของ DQN ใช้ค่าจาก `--dqn_merge_sizes` ถ้าไม่ใส่จะใช้ `--merge_sizes` ของ COMI
  ควรใช้ชุดเดียวกับที่ COMI ถูก train มา (COMI default คือ `16,32`) เพราะถ้า COMI ไม่เคยเห็น merge_size ไหน คุณภาพที่ค่านั้นจะตก
- ช่วง train ใช้ epsilon-greedy โดย `--dqn_epsilon_decay_steps 0` = 70% ของ train_size × epochs (เหมือนต้นฉบับ)
- ถ้ามี `--dqn_checkpoint` อยู่แล้ว ตอน train จะโหลดมาเทรนต่อ (มีทั้ง replay และ epsilon)
- รันได้ทีละ 1 process ยังไม่รองรับ torchrun หลาย GPU

ไฟล์ output ใน `--output_dir`
- train: `dqn_controller.pt`, `training_history.json` (reward / F1 / saving / loss / epsilon ต่อ epoch),
  `train_epoch{N}.jsonl`, `run_config.json`
- eval: `comparison.json` (DQN เทียบกับ Fixed-m), `eval_dqn.jsonl`, `eval_fixed_{m}.jsonl`

## ใช้ใน Python

```python
from comi_dqn import DQNOverCOMI, summarize
from prompt_controller import PromptController, TokenEncoder, EncoderConfig, actions_from_merge_sizes

ctl = PromptController(TokenEncoder(EncoderConfig('hf')), actions_from_merge_sizes('16,32'))
runner = DQNOverCOMI(comi_model, ctl)                 # comi_model = COMI ที่โหลดแล้ว
runner.run(train_samples, explore=True, train=True)   # train DQN
print(summarize(runner.run(test_samples)))            # greedy
ctl.save('dqn_controller.pt')
```

## ใช้ controller อย่างเดียว (ไม่โหลด COMI)

ถ้าต้องการรัน DQN กับ COMI แยกกันคนละขั้น ใช้ CLI ของ `prompt_controller.py` (init / decide / train)
ตามหัวข้อข้างล่าง

## State (7 ค่า)

`[input_tokens/4096 (cap 2), mean relevance, max relevance, mean MIG, max MIG, MIG std, mean redundancy]`

- relevance = cos(word, mean(query))
- redundancy = cos สูงสุดกับคำอื่นใน context (ถ้ายาวเกิน 512 คำ ใช้ anchor 512 ตัว)
- MIG = relevance − redundancy

## Action

ค่า default เหมือนต้นฉบับ:

| index | name | retention | compress_rate |
|---|---|---|---|
| 0 | NONE | 1.00 | 1 |
| 1 | LIGHT | 0.75 | 1.333 |
| 2 | MEDIUM | 0.50 | 2 |
| 3 | STRONG | 0.25 | 4 |

COMI ใช้ `merge_size` เป็นจำนวนเต็ม ถ้าจะให้ action ตรงกับ COMI ให้สร้าง checkpoint ด้วย `--merge-sizes`
เช่น `--merge-sizes 1,4,16,32` (1 = ไม่บีบ) decision จะมีค่า `merge_size` ให้ใช้ได้ตรงๆ

## Reward

```
reward = 0.55·quality + 0.20·token_saving + 0.15·energy_saving + 0.10·latency_saving
         − 1.0 ถ้า quality < 0.70
token_saving = 1 − compressed_tokens / original_tokens
```

ค่า `--cost-source tokens` (default) จะใช้ token_saving แทน energy และ latency
ผลคือ `0.55·quality + 0.45·token_saving − penalty`
ถ้าเลือก `measured` ต้องส่ง `latency_saving` และ `energy_saving` จาก COMI มาด้วย

## CLI ของ controller (แยกขั้นกับ COMI)

```bash
pip install -r requirements.txt

# 1) สร้าง controller
python prompt_controller.py init --checkpoint runs/ctl.pt \
    --encoder-mode hf --merge-sizes 1,4,16,32 --epsilon-decay-steps 7000

# 2) เลือก action ให้ request  (ช่วงเก็บข้อมูล train ใส่ --explore)
python prompt_controller.py decide --checkpoint runs/ctl.pt \
    --input data/requests.jsonl --output runs/decisions.jsonl --explore

# 3) COMI อ่าน decisions.jsonl บีบตาม merge_size แล้วเขียน feedback.jsonl

# 4) train จาก feedback
python prompt_controller.py train --checkpoint runs/ctl.pt --feedback runs/feedback.jsonl
```

ทำ 2 → 4 วนไปหลายรอบ checkpoint จะเก็บ replay buffer และ `env_steps` ไว้ด้วย epsilon จึงลดต่อเนื่องข้ามรอบ
ตอน evaluate ให้รัน `decide` โดยไม่ใส่ `--explore` (เลือกแบบ greedy)

### รูปแบบไฟล์

`requests.jsonl`
```json
{"request_id": "1", "context": "...", "query": "..."}
```

`decisions.jsonl` (output ของ controller)
```json
{"request_id": "1", "action_index": 2, "action_name": "MERGE_16", "retention_ratio": 0.0625,
 "compress_rate": 16.0, "merge_size": 16, "input_tokens": 2048, "target_tokens": 128,
 "state": [0.5, 0.31, 0.92, -0.42, 0.11, 0.18, 0.73], "q_values": [...],
 "explored": false, "epsilon": null, "feature_version": "prompt-word-r-minus-d-v1"}
```

`feedback.jsonl` = แถวของ decision เดิม แล้วเติมผลที่ COMI วัดได้
```json
{"...ทุก field ของ decision...", "quality": 0.82, "compressed_tokens": 128, "original_tokens": 2310}
```
- `quality` อยู่ในช่วง [0, 1] เช่น F1/EM ของคำตอบเทียบ reference
- `compressed_tokens` คือจำนวน memory slot หลังบีบที่ COMI ได้จริง
- `original_tokens` (ไม่บังคับ) คือจำนวน token ก่อนบีบที่ COMI นับเอง ถ้าไม่ส่งมาจะใช้ `input_tokens` (นับแบบ word) แทน
  ควรส่งมา เพราะ COMI นับเป็น subword ต่อ subword จะได้ token_saving ที่ถูกต้อง

controller จะ error ถ้า action table หรือ feature_version ของ feedback ไม่ตรงกับ checkpoint

## Test

```bash
python tests/test_controller.py
python tests/test_comi_wrapper.py
```

`test_comi_wrapper.py` ใช้ COMI ปลอมที่มี interface เดียวกัน ตรวจว่าเรียงลำดับผลถูก, คืนค่า merge_size/is_random ให้ COMI,
ถอด hook ออกครบ, นับ memory slot ถูก, แบ่ง batch ตาม merge_size และ DQN ชนะ fixed policy ทุกค่า

- state ตรงกับการคำนวณแบบ brute-force และไม่พังเมื่อ context ว่าง, มีคำเดียว หรือยาวเกิน 512 คำ
- สูตร reward ตรงตัวอย่าง (2048 → 128 token, quality 0.8 → 0.862)
- learnability: ข้อมูลสังเคราะห์ 3 แบบที่ action ที่ดีที่สุดต่างกัน DQN ต้องชนะ fixed policy ทุกแบบ
  ผลล่าสุด DQN = 0.735 / NONE = 0.550 / MEDIUM = 0.313 / LIGHT = 0.200 / STRONG = −0.037 (ค่าที่ดีที่สุดเป็นไปได้ = 0.738)
- save/load checkpoint ได้ decision เท่าเดิม และ replay กับ env_steps ไม่หาย
- CLI ทำงานครบ init → decide → train

## ข้อควรรู้

- ทุก request มี decision เดียวและจบในตัว (`done=True`, `gamma=0`) ปัญหานี้คือ contextual bandit ที่ใช้โครงสร้าง DQN
  target network ไม่มีผลกับผลลัพธ์ในการตั้งค่านี้
- state คำนวณจาก text ด้วย encoder ของ controller เอง (MiniLM หรือ hash) ไม่ได้ใช้ hidden state ของ COMI
  ค่าจึงไม่เท่ากับ state ใน `dqn_controller.py` ของ branch `omar/controller_dev` ที่คำนวณจาก hidden state ของ encoder ใน COMI
- `hash` mode เป็นเวกเตอร์สุ่ม ใช้ทดสอบ pipeline เท่านั้น ตอนทดลองจริงต้องใช้ `--encoder-mode hf`
- `run_dqn_over_comi.py` ทดสอบกับโค้ด `modeling_comi.py` จริงโดยใช้ Qwen2 ขนาดเล็กแบบสุ่ม weight ผ่านทั้ง train และ eval
  แต่ยังไม่ได้รันกับ COMI checkpoint ที่ train แล้ว ตัวเลข F1 จริงต้องรันบน GPU
