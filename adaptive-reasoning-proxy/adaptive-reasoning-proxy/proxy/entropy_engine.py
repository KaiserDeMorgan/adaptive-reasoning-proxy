import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class EntropyState:
    window: deque = field(default_factory=lambda: deque(maxlen=20))
    ema: float = 0.0
    alpha: float = 0.15
    tokens_generated: int = 0
    stopped_early: bool = False

    def update(self, logprobs: dict[str, float]) -> float:
        """
        Compute H(t) = -sum(p * log p) for one token position.

        Args:
            logprobs: mapping of token string -> log probability (top-k from API)

        Returns:
            Shannon entropy at this token position
        """
        probs = np.array([np.exp(lp) for lp in logprobs.values()])
        probs /= probs.sum()
        h = float(-np.sum(probs * np.log(probs + 1e-12)))
        self.window.append(h)
        self.ema = self.alpha * h + (1 - self.alpha) * self.ema
        self.tokens_generated += 1
        return h

    def should_stop(self, threshold: float, min_tokens: int = 40) -> bool:
        """
        The EDRM insight: sustained LOW entropy = model has settled into a
        low-entropy 'reasoning regime'. Stop here rather than generating
        redundant confirmatory tokens.

        Args:
            threshold: entropy value below which generation is considered settled
            min_tokens: minimum tokens before early stop is allowed

        Returns:
            True if generation should stop early
        """
        if self.tokens_generated < min_tokens:
            return False
        if len(self.window) < 10:
            return False
        recent_mean = np.mean(list(self.window)[-10:])
        return bool(recent_mean < threshold and self.ema < threshold * 1.1)
