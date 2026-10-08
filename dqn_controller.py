"""COMI adapter: encoder embeddings -> seven features -> discrete merge size."""
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from dqn_agent import DQNAgent, DQNConfig


class DQNMergeController:
    # Intentionally not nn.Module: COMI's optimizer/state_dict must not own DQN.
    FEATURE_VERSION = 'comi-hidden-token-r-minus-d-v1'

    def __init__(self, agent, merge_sizes=(16, 32), explore=False,
                 max_pairwise=512, normalize_length_by=4096.0):
        self.merge_sizes = tuple(merge_sizes)
        if not self.merge_sizes or any(type(x) is not int or x <= 0 for x in self.merge_sizes):
            raise ValueError('merge_sizes must contain positive integers')
        if len(set(self.merge_sizes)) != len(self.merge_sizes):
            raise ValueError('merge_sizes must be unique; order defines action meanings')
        if agent.config.state_dim != 7 or agent.config.action_dim != len(self.merge_sizes):
            raise ValueError('DQN dimensions must match seven features and merge_sizes')
        if type(max_pairwise) is not int or max_pairwise < 2 or normalize_length_by <= 0:
            raise ValueError('Invalid feature normalization or anchor count')
        self.agent = agent
        self.explore = bool(explore)
        self.max_pairwise = max_pairwise
        self.normalize_length_by = float(normalize_length_by)

    @torch.no_grad()
    def build_state(self, context_embeddings, query_embedding):
        # Detach only the feature branch. COMI still compresses original tensors.
        context = context_embeddings.detach().float()
        query = query_embedding.detach().float()
        n = context.shape[0]
        if n == 0:
            return np.zeros(7, dtype=np.float32)
        context = F.normalize(context, dim=-1)
        query = F.normalize(query, dim=-1)
        relevance = context @ query
        if n == 1:
            redundancy = torch.zeros_like(relevance)
        else:
            indices = torch.linspace(0, n - 1, min(n, self.max_pairwise),
                                     device=context.device).long()
            anchors = context[indices]
            chunks = []
            # Bound temporary similarity storage to 1024 x max_pairwise.
            for start in range(0, n, 1024):
                stop = min(start + 1024, n)
                similarity = context[start:stop] @ anchors.T
                rows = torch.arange(start, stop, device=context.device)
                similarity.masked_fill_(rows[:, None] == indices[None, :], -torch.inf)
                chunks.append(similarity.max(dim=1).values)
            redundancy = torch.cat(chunks)
        mig = relevance - redundancy
        state = torch.stack((
            relevance.new_tensor(min(n / self.normalize_length_by, 2.0)),
            relevance.mean(), relevance.max(), mig.mean(), mig.max(),
            mig.std(unbiased=False), redundancy.mean(),
        ))
        if not torch.isfinite(state).all():
            raise ValueError('Non-finite DQN state from COMI embeddings')
        return state.cpu().numpy()

    def choose(self, context_embeddings, query_embedding):
        state = self.build_state(context_embeddings, query_embedding)
        action = self.agent.select_action(state, training=self.explore)
        return self.merge_sizes[action], {'state': state.copy(), 'action': action}

    def save(self, path):
        # Includes feature and action semantics, unlike the generic agent checkpoint.
        torch.save({
            'feature_version': self.FEATURE_VERSION,
            'merge_sizes': list(self.merge_sizes),
            'max_pairwise': self.max_pairwise,
            'normalize_length_by': self.normalize_length_by,
            'config': asdict(self.agent.config),
            'policy': self.agent.policy_net.state_dict(),
            'target': self.agent.target_net.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'steps': self.agent.total_steps, 'env_steps': self.agent.env_steps,
        }, path)

    @classmethod
    def from_checkpoint(cls, path, device='cpu', explore=False):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get('feature_version') != cls.FEATURE_VERSION:
            raise ValueError('Checkpoint must use this COMI feature definition')
        agent = DQNAgent(DQNConfig(**payload['config']), device=device)
        controller = cls(agent, payload['merge_sizes'], explore,
                         payload['max_pairwise'], payload['normalize_length_by'])
        agent.policy_net.load_state_dict(payload['policy'])
        agent.target_net.load_state_dict(payload['target'])
        agent.optimizer.load_state_dict(payload['optimizer'])
        agent.total_steps = int(payload['steps'])
        agent.env_steps = int(payload['env_steps'])
        return controller
