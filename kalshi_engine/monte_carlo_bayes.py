from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        d = self.alpha + self.beta
        return 0.5 if d <= 0 else self.alpha / d


def monte_carlo_probability(distribution_fn: Callable[[int], np.ndarray], n: int = 100_000) -> float:
    """
    distribution_fn must return boolean-like array indicating YES outcomes.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    samples = np.asarray(distribution_fn(int(n)))
    if samples.size == 0:
        raise ValueError("distribution_fn returned empty samples")
    return float(np.mean(samples.astype(float)))


def beta_update(prior_alpha: float, prior_beta: float, successes: int, failures: int) -> BetaPosterior:
    a0 = max(1e-9, float(prior_alpha))
    b0 = max(1e-9, float(prior_beta))
    s = max(0, int(successes))
    f = max(0, int(failures))
    return BetaPosterior(alpha=a0 + s, beta=b0 + f)


def bayesian_probability_from_stream(
    prior_alpha: float,
    prior_beta: float,
    observations: np.ndarray,
) -> BetaPosterior:
    """
    observations is array-like of 0/1 labels.
    """
    obs = np.asarray(observations).astype(int)
    s = int(np.sum(obs == 1))
    f = int(np.sum(obs == 0))
    return beta_update(prior_alpha, prior_beta, s, f)
