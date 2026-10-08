"""CPU checks of actual COMI methods with a tiny encoder; no HF downloads."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn_agent import DQNAgent, DQNConfig
from dqn_controller import DQNMergeController

# Load actual class bodies while avoiding unrelated Hugging Face imports/loaders.
source = ast.parse(Path(__file__).with_name('modeling_comi.py').read_text())
classes = [n for n in source.body if isinstance(n, ast.ClassDef)
           and n.name in {'GroupReallocator', 'TKDR', 'COMI'}]
namespace = dict(torch=torch, nn=nn, F=F, math=math)
exec(compile(ast.Module(body=classes, type_ignores=[]), 'modeling_comi.py', 'exec'), namespace)
COMI, TKDR = namespace['COMI'], namespace['TKDR']

class TinyEncoder(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(hidden_states=[inputs_embeds * 1.2])

class TinyDecoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.config = SimpleNamespace(hidden_size=8)
    def get_input_embeddings(self):
        return self.embedding

def fixture():
    model = COMI.__new__(COMI)
    nn.Module.__init__(model)
    embedding = nn.Embedding(40, 8)
    model.encoder = TinyEncoder()
    model.decoder = TinyDecoder(embedding)
    model.encoder.model = nn.Module()
    model.encoder.model.embed_tokens = embedding
    model.merge_size = 2
    model.merge_sizes = [2, 4]
    model.segment_size = 6
    model.is_random = False
    model.use_transform_layer = False
    model.dqn_controller = None
    model.last_dqn_decisions = []
    model.tkdr = TKDR(1., 1., 1., 1., coarse_grained_on=False)
    return model

def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    agent = DQNAgent(DQNConfig(action_dim=2, batch_size=2, warmup_steps=2), device='cpu')
    controller = DQNMergeController(agent, merge_sizes=(2, 4))
    # Force greedy action 1 so exact expected merge sizes are known.
    with torch.no_grad():
        for p in agent.policy_net.parameters(): p.zero_()
        agent.policy_net.net[-1].bias[1] = 1.
    for n in (0, 1, 10, 513, 1100):
        state = controller.build_state(torch.randn(n, 8), torch.randn(8))
        assert state.shape == (7,) and np.isfinite(state).all()
    # Compare state implementation against direct all-pairs computation.
    c, q = torch.randn(10, 8), torch.randn(8)
    cn, qn = F.normalize(c, dim=-1), F.normalize(q, dim=-1)
    r = cn @ qn
    sim = cn @ cn.T
    sim.fill_diagonal_(-torch.inf)
    d = sim.max(1).values
    m = r-d
    expected = np.array([10/4096, r.mean(), r.max(), m.mean(), m.max(), m.std(unbiased=False), d.mean()])
    np.testing.assert_allclose(controller.build_state(c, q), expected, atol=1e-6)
    model = fixture()
    ids = torch.randint(1, 40, (2, 9))
    mask = torch.tensor([[1]*9, [1]*4+[0]*5])
    query = torch.ones((2, 2), dtype=torch.long)
    query_mask = torch.ones_like(query)
    baseline = model.build_comi_memory(ids, mask, query, query_mask)
    model.set_dqn_controller(controller)
    model.is_random = True
    model.generate_merge_size = lambda: (_ for _ in ()).throw(AssertionError('random override'))
    memory, memory_mask = model.build_comi_memory(ids, mask, query, query_mask)
    assert len(model.last_dqn_decisions) == 3  # Skip padded-only sample/segment.
    assert all(d['merge_size'] == 4 for d in model.last_dqn_decisions)
    assert {(d['batch_index'], d['segment_index']) for d in model.last_dqn_decisions} == {(0,0),(1,0),(0,1)}
    assert memory_mask.dtype == torch.bool and memory.shape[:2] == memory_mask.shape
    memory.square().sum().backward()
    assert model.decoder.embedding.weight.grad is not None
    assert all(p.grad is None for p in agent.policy_net.parameters())
    model.build_comi_memory(ids, mask, query, query_mask)
    assert len(model.last_dqn_decisions) == 3  # Log resets per request.
    model.set_dqn_controller(None)
    model.is_random = False
    restored = model.build_comi_memory(ids, mask, query, query_mask)
    torch.testing.assert_close(baseline[0], restored[0])
    torch.testing.assert_close(baseline[1], restored[1])
    assert not any('policy_net' in name for name in model.state_dict())
    # Exercise a genuine DQN optimizer update and separate checkpoint round trip.
    for reward in (0.5, -0.2):
        agent.remember(state, 1, reward, np.zeros(7, np.float32), True)
    assert math.isfinite(agent.train_step())
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'controller.pt'
        controller.save(path)
        loaded = DQNMergeController.from_checkpoint(path)
        assert loaded.merge_sizes == (2,4)
        for p, other in zip(agent.policy_net.parameters(), loaded.agent.policy_net.parameters()):
            torch.testing.assert_close(p, other)
    # Existing coarse reallocation still receives the chosen rate.
    model.set_dqn_controller(controller)
    model.tkdr.coarse_grained_on = True
    memory, _ = model.build_comi_memory(ids, mask, query, query_mask)
    assert torch.isfinite(memory).all()
    print('PASS: features, selection, padding, segments, random precedence, gradient isolation, disabled parity, replay update, checkpoint, coarse compression')

if __name__ == '__main__':
    main()
