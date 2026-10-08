"""DQN reward from response quality and normalized context-token savings only."""
from dataclasses import dataclass
import math


@dataclass
class RewardConfig:
    quality_weight: float = 0.55
    token_weight: float = 0.45
    quality_min: float = 0.70
    quality_penalty: float = 1.0


class RewardCalculator:
    def __init__(self, config=None):
        self.config = config if config is not None else RewardConfig()

    def calculate(self, quality, token_saving):
        """quality in [0,1]; token_saving is a fraction, not a raw count."""
        quality, token_saving = float(quality), float(token_saving)
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError('quality must be finite and in [0, 1]')
        if not math.isfinite(token_saving) or not -1 <= token_saving <= 1:
            raise ValueError('token_saving must be finite and in [-1, 1]')
        c = self.config
        reward = c.quality_weight * quality + c.token_weight * token_saving
        if quality < c.quality_min:
            reward -= c.quality_penalty
        return float(reward)

    def from_counts(self, quality, original_tokens, compressed_tokens):
        """Use valid context tokens before compression and actual memory slots after."""
        original_tokens, compressed_tokens = float(original_tokens), float(compressed_tokens)
        if (not math.isfinite(original_tokens) or original_tokens <= 0
                or not original_tokens.is_integer()):
            raise ValueError('original_tokens must be a positive integer count')
        if (not math.isfinite(compressed_tokens) or compressed_tokens < 0
                or not compressed_tokens.is_integer()):
            raise ValueError('compressed_tokens must be a nonnegative integer count')
        saving = max(-1.0, min(1.0, 1.0 - compressed_tokens / original_tokens))
        return self.calculate(quality, saving)
