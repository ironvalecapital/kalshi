from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BayesResult:
    prior: float
    implied: float
    posterior: float
    fair_value_cents: float
    edge_cents: float


def bayes_update(prior: float, implied: float, prior_weight: float = 4.0) -> BayesResult:
    posterior = (prior_weight * prior + implied) / (prior_weight + 1.0)
    fair_value = posterior * 100.0
    edge = fair_value - implied * 100.0
    return BayesResult(prior=prior, implied=implied, posterior=posterior, fair_value_cents=fair_value, edge_cents=edge)


def choose_prior(prior_override: Optional[float] = None) -> float:
    if prior_override is not None:
        return max(0.01, min(0.99, prior_override))
    return 0.5
