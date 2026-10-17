"""Offline CPU smoke tests for the DQN training and inference pipelines.

Run from the repository root:
    python -m unittest discover -s omar/controller_over_COMI/tests -v

Uses real DQN optimization/checkpoints and a fake COMI generator. Requires
numpy, torch and tqdm; no model downloads, GPU, PEFT or Transformers needed.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dqn_over_comi_infer as inference
import dqn_over_comi_train as training
from prompt_controller import PromptController, set_seed


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        lengths = [len(text.split()) for text in texts]
        mask = torch.tensor([[1] * n + [0] * (max(lengths) - n) for n in lengths])
        return {'input_ids': mask.clone(), 'attention_mask': mask}

    def batch_decode(self, outputs, **kwargs):
        return outputs


class FakeCOMI(torch.nn.Module):
    """Keep COMI's generation and memory-mask interface, without an LLM."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.tokenizer = FakeTokenizer()
        self.merge_size, self.is_random = 16, True
        self.eval()

    def build_comi_memory(self, input_ids, attention_mask, query_ids, query_attention_mask):
        slots = [math.ceil(int(n) / self.merge_size) for n in attention_mask.sum(1)]
        mask = torch.tensor([[1] * n + [0] * (max(slots) - n) for n in slots], dtype=torch.bool)
        return torch.zeros(len(slots), max(slots), 4), mask

    def forward(self, input_ids, attention_mask, query_ids, query_attention_mask, labels=None):
        assert labels is None and not self.is_random
        assert not torch.is_grad_enabled(), 'COMI generation must run without gradients'
        self.build_comi_memory(input_ids, attention_mask, query_ids, query_attention_mask)
        return ['Paris'] * len(input_ids)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class DQNPipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        set_seed(7)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name)
        self.model_args = SimpleNamespace(model_name_or_path='fake-comi', merge_sizes='4,8,16')
        self.training_args = SimpleNamespace(output_dir=str(self.output_dir), restore_from='')
        self.dqn_args = training.DQNTrainingArguments(
            dqn_encoder_mode='hash', dqn_epochs=1, dqn_batch_size=7)
        # Cross the default replay warmup (32) so smoke tests prove that actual
        # backpropagation happens, and include a final partial batch.
        self.samples = [
            {'input': ' '.join(['Paris', 'France'] * (5 + i % 4)),
             'prompt': 'What is the capital of France?',
             'answer': 'Paris' if i % 2 else ['London', 'The Paris.'], 'sample_id': i}
            for i in range(36)
        ]
        data_path = self.output_dir / 'samples.jsonl'
        training.write_jsonl(data_path, self.samples)
        self.samples = training.load_samples(data_path, shuffle=True, seed=7)

    def prepare_training(self):
        with patch.object(training, 'prepare_model', return_value=FakeCOMI()) as loader:
            runner, checkpoint = training.prepare_dqn_model(
                self.model_args, self.training_args, self.dqn_args, torch.device('cpu'), len(self.samples))
        loader.assert_called_once()
        return runner, checkpoint

    def train(self, runner, checkpoint):
        return training.train_dqn(runner, self.samples, self.model_args,
                                  self.training_args, self.dqn_args, checkpoint)

    def test_train_saves_updated_controller_and_artifacts(self):
        runner, checkpoint = self.prepare_training()
        before = {k: v.clone() for k, v in runner.controller.agent.policy_net.state_dict().items()}
        history = self.train(runner, checkpoint)
        agent = runner.controller.agent
        self.assertEqual(agent.env_steps, 36)
        self.assertEqual(len(agent.replay), 36)
        self.assertGreater(agent.total_steps, 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in agent.policy_net.state_dict().items()))
        self.assertTrue(math.isfinite(history[0]['loss_mean']))
        self.assertEqual(history[0]['n'], 36)
        self.assertEqual(json.loads((self.output_dir / 'training_history.json').read_text()), history)
        config = json.loads((self.output_dir / 'run_config.json').read_text())
        self.assertEqual(config['encoder']['mode'], 'hash')
        self.assertEqual(config['merge_sizes'], [4, 8, 16])
        rows = read_jsonl(self.output_dir / 'train_epoch1.jsonl')
        self.assertEqual(sorted(row['sample_id'] for row in rows), list(range(36)))
        for row in rows:
            self.assertEqual(row['compressed_tokens'], math.ceil(row['original_tokens'] / row['merge_size']))
            self.assertTrue(math.isfinite(row['reward']))
        loaded = PromptController.load(checkpoint, device='cpu')
        self.assertEqual(loaded.agent.total_steps, agent.total_steps)
        self.assertEqual(len(loaded.agent.replay), 36)
        self.assertEqual(runner.comi.merge_size, 16)
        self.assertTrue(runner.comi.is_random)
        self.assertNotIn('build_comi_memory', runner.comi.__dict__)
        self.assertIsNone(runner.comi.weight.grad)

    def test_train_resumes_checkpoint(self):
        runner, checkpoint = self.prepare_training()
        self.train(runner, checkpoint)
        previous_steps = runner.controller.agent.total_steps
        # A resumed controller must retain its saved encoder and action table.
        self.dqn_args.dqn_encoder_mode = 'hf'
        self.dqn_args.dqn_merge_sizes = '2,32'
        resumed, resumed_checkpoint = self.prepare_training()
        self.assertEqual(resumed_checkpoint, checkpoint)
        self.assertEqual(resumed.controller.encoder.config.mode, 'hash')
        self.assertEqual([a.compress_rate for a in resumed.controller.actions], [4, 8, 16])
        self.assertEqual(resumed.controller.agent.env_steps, 36)
        self.assertEqual(resumed.controller.agent.total_steps, previous_steps)
        self.assertTrue(resumed.controller.agent.optimizer.state_dict()['state'])
        self.train(resumed, checkpoint)
        loaded = PromptController.load(checkpoint, device='cpu')
        self.assertEqual(loaded.agent.env_steps, 72)
        self.assertEqual(len(loaded.agent.replay), 72)
        self.assertGreater(loaded.agent.total_steps, previous_steps)

    def test_inference_loads_checkpoint_writes_predictions_and_preserves_policy(self):
        runner, checkpoint = self.prepare_training()
        self.train(runner, checkpoint)
        with patch.object(inference, 'prepare_model', return_value=FakeCOMI()):
            model = inference.prepare_dqn_model(
                self.model_args, self.training_args, inference.DQNArguments(), torch.device('cpu'))
        agent = model.controller.agent
        before = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
        counters = (agent.env_steps, agent.total_steps, len(agent.replay))
        samples = inference.load_eval_samples(self.output_dir / 'samples.jsonl', 9, False, 7)
        output_path, _, tmp_dir = inference.make_output_paths(self.output_dir, 9, 'dqn')
        # Simulate two inference ranks sequentially on CPU, including uneven
        # shards and a partial batch, then exercise the real merge/metric code.
        for rank in range(2):
            rows = inference.infer_rank_batches(model, samples, 4, self.training_args,
                                                torch.device('cpu'), rank, 2)
            self.assertEqual([r['sample_id'] for r in rows], list(range(rank, 9, 2)))
            inference.write_rank_results(rows, tmp_dir, rank)
        merged = inference.merge_rank_outputs(tmp_dir, 2, output_path)
        self.assertEqual(read_jsonl(output_path), merged)
        self.assertEqual([row['sample_id'] for row in merged], list(range(9)))
        self.assertEqual(inference.compute_metrics(merged), {'total_samples': 9, 'avg_em': 1.0, 'avg_f1': 1.0})
        for row, sample in zip(merged, samples):
            self.assertEqual(row['context'], sample['input'])
            self.assertEqual(row['question'], sample['prompt'])
            self.assertFalse(row['explored'])
            self.assertNotIn('loss', row)
        self.assertEqual((agent.env_steps, agent.total_steps, len(agent.replay)), counters)
        for name, value in agent.policy_net.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        expected = runner.run(samples, batch_size=4)
        self.assertEqual([r['merge_size'] for r in merged], [r['merge_size'] for r in expected])


if __name__ == '__main__':
    unittest.main()
