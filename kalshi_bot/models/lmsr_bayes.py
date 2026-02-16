from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple


def _clamp_prob(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def lmsr_yes_probability(depth_yes: int, depth_no: int, b: float) -> float:
    """
    Approximate LMSR yes probability from per-side depth.
    We treat depth as pseudo-share imbalance and convert via softmax.
    """
    depth_yes = max(0, int(depth_yes))
    depth_no = max(0, int(depth_no))
    b = max(1e-6, float(b))
    x = (depth_yes - depth_no) / b
    if x >= 0:
        z = math.exp(-x)
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        p = z / (1.0 + z)
    return _clamp_prob(p)


@dataclass
class BayesianPriceResult:
    posterior_yes: float
    alpha: float
    beta: float


def bayesian_yes_probability(
    prior_yes: float,
    prior_weight: float,
    evidences: Iterable[Tuple[float, float]],
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> BayesianPriceResult:
    """
    Beta-Bernoulli style pricing blend:
    - prior_yes contributes with prior_weight pseudo-observations
    - each evidence contributes weight pseudo-observations at probability p
    """
    prior = _clamp_prob(prior_yes)
    alpha = float(alpha0) + max(0.0, float(prior_weight)) * prior
    beta = float(beta0) + max(0.0, float(prior_weight)) * (1.0 - prior)

    for p, w in evidences:
        w = max(0.0, float(w))
        p = _clamp_prob(p)
        alpha += w * p
        beta += w * (1.0 - p)

    denom = alpha + beta
    post = prior if denom <= 0 else alpha / denom
    return BayesianPriceResult(posterior_yes=_clamp_prob(post), alpha=alpha, beta=beta)
