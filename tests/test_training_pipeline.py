import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from prepare_hotpotqa import convert_row, export_split
from scripts.preflight import validate_file
from training_utils import DataCollatorForDynamicPadding, InstructFTTokenizeFunction, train_model
from modeling_comi import COMI, ModelArguments, TrainingArguments, build_lora_config
from ft_inference import prepare_model
from peft import LoraConfig


def test_hotpot_conversion_keeps_distractors(tmp_path):
    row = {"id": "example", "question": "Where?", "answer": "Paris",
           "context": {"title": ["Relevant", "Distractor"],
                       "sentences": [[" Paris is here. "], ["Keep this too."]]},
           "supporting_facts": {"title": ["Relevant"], "sent_id": [0]}}
    result = convert_row(row)
    assert result["input"] == "Relevant\nParis is here.\n\nDistractor\nKeep this too."
    assert result["answer"] == ["Paris"]
    path = tmp_path / "train.jsonl"
    stats = export_split([row], path)
    assert stats["rows"] == validate_file(path) == 1
    path.write_text(json.dumps({"input": "a", "prompt": "b", "answer": []}) + "\n")
    with pytest.raises(ValueError, match="answer"):
        validate_file(path)


def test_padding_preserves_real_eos():
    batch = DataCollatorForDynamicPadding(0)([
        {"input_ids": [1, 0], "query_ids": [2], "labels": [3, 0]},
        {"input_ids": [1], "query_ids": [2, 0], "labels": [3]},
    ])
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert batch["query_attention_mask"].tolist() == [[True, False], [True, True]]
    assert batch["labels"].tolist() == [[3, 0], [3, -100]]


@pytest.mark.parametrize("all_lora", [False, True])
def test_tiny_qwen_train_checkpoint_resume_generate(tmp_path, all_lora):
    torch.set_num_threads(1)
    base = tmp_path / "base"
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "a": 3, "b": 4, "c": 5}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                  eos_token="[EOS]", unk_token="[UNK]")
    fast.save_pretrained(base)
    config = Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
                        pad_token_id=0, eos_token_id=1, tie_word_embeddings=True)
    Qwen3ForCausalLM(config).save_pretrained(base)
    model_args = ModelArguments(model_name_or_path=str(base), encoder_layers=1,
                                num_mem_fusion_layers=1, merge_size=2, segment_size=4,
                                lora_r=2, lora_alpha=4, train=True, max_new_tokens=3)
    if all_lora:
        model_args.full_finetune_encoder = False
        model_args.full_finetune_lsa = False
        model_args.lora_encoder = True
        model_args.lora_lsa = True
    output = tmp_path / "output"
    args = TrainingArguments(output_dir=str(output), use_cpu=True, bf16=False,
                             max_steps=1, save_steps=1, eval_steps=1, warmup_ratio=0,
                             logging_steps=1, report_to="none", gradient_checkpointing=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             per_device_eval_batch_size=1, prediction_loss_only=True)
    model = COMI(model_args, args, LoraConfig(**build_lora_config(model_args)))
    raw_data = Dataset.from_list([
        {"input": "a b c a b", "prompt": "a b", "answer": ["c"]},
        {"input": "a c", "prompt": "b", "answer": ["a"]},
    ])
    data = raw_data.map(InstructFTTokenizeFunction(fast), batched=True,
                        remove_columns=raw_data.column_names)
    tracked_weight = next(parameter for name, parameter in model.encoder.named_parameters()
                          if ("lora_B" in name if all_lora else "self_attn.q_proj.weight" in name))
    before = tracked_weight.detach().clone()
    train_model(model, data, data, args, fast)
    assert not torch.equal(before, tracked_weight)
    assert (output / "checkpoint-1" / "optimizer.pt").exists()
    assert json.loads((output / "comi_model_args.json").read_text())["encoder_layers"] == 1
    eval_args = TrainingArguments(output_dir=str(tmp_path / "eval"), use_cpu=True,
                                  bf16=False, restore_from=str(output))
    # Deliberately wrong layer count: checkpoint metadata must restore the architecture.
    inference_args = ModelArguments(model_name_or_path=str(base), encoder_layers=2,
                                    merge_size=2, segment_size=4, max_new_tokens=3)
    restored = prepare_model(inference_args, eval_args, torch.device("cpu"))
    encoder_base = restored.encoder.get_base_model() if all_lora else restored.encoder
    assert len(encoder_base.model.layers) == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with torch.inference_mode():
        batch = DataCollatorForDynamicPadding(0)([data[0]])
        batch.pop("labels")
        generated = restored(**batch)
    assert generated.shape[0] == 1 and generated.shape[1] <= 3
    args.max_steps = 2
    resumed = COMI(model_args, args, LoraConfig(**build_lora_config(model_args)))
    train_model(resumed, data, data, args, fast)
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 2
