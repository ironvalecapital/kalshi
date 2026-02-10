from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConsistencyResult:
    max_diff: float
    consistent: bool


def check_consistency(implied_probs: List[float], threshold: float = 0.15) -> ConsistencyResult:
    if not implied_probs:
        return ConsistencyResult(max_diff=0.0, consistent=False)
    max_p = max(implied_probs)
    min_p = min(implied_probs)
    diff = max_p - min_p
    return ConsistencyResult(max_diff=diff, consistent=diff <= threshold)


def maybe_abstain(result: ConsistencyResult) -> Optional[str]:
    if not result.consistent:
        return "abstain"
    return None
