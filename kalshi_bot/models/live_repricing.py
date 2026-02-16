from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class LiveState:
    score_diff: int  # team_of_interest - opponent
    time_remaining_sec: int
    possession: int  # +1 team_of_interest, -1 opponent, 0 unknown
    efficiency_edge: float  # points-per-possession edge (team - opp)
    pace: float  # possessions per 48 min


def _clamp(min_v: float, v: float, max_v: float) -> float:
    return max(min_v, min(max_v, v))


def _logit_prob(x: float) -> float:
    # Numerically stable sigmoid to avoid overflow at extreme logits.
    if x >= 0.0:
        z = math.exp(-x)
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        p = z / (1.0 + z)
    return _clamp(1e-12, p, 1.0 - 1e-12)


def _sample_points(rng: random.Random, ppp: float) -> int:
    """
    Sample possession points from a compact distribution calibrated to PPP.
    """
    ppp = _clamp(0.7, ppp, 1.45)

    # Simple scoring mixture; 2PT mass is solved from target PPP.
    p1 = 0.08
    p3 = 0.10
    p2 = _clamp(0.18, (ppp - (p1 + 3.0 * p3)) / 2.0, 0.60)
    p0 = _clamp(0.02, 1.0 - (p1 + p2 + p3), 0.74)
    norm = p0 + p1 + p2 + p3
    p0, p1, p2, p3 = p0 / norm, p1 / norm, p2 / norm, p3 / norm

    u = rng.random()
    if u < p0:
        return 0
    if u < p0 + p1:
        return 1
    if u < p0 + p1 + p2:
        return 2
    return 3


def win_probability(
    score_diff: int,
    time_remaining_sec: int,
    possession: int,
    efficiency_edge: float,
    pace: float,
) -> float:
    """
    Fast deterministic approximation:
    P(win) = sigmoid(a*score_diff + b*poss_left*eff_edge + c*possession + d*time_decay*score_diff)
    """
    t = max(0, int(time_remaining_sec))
    poss_left = max(0.0, float(pace) * (t / 2880.0))
    time_decay = 1.0 - _clamp(0.0, t / 2880.0, 1.0)
    x = (
        0.075 * float(score_diff)
        + 0.60 * poss_left * float(efficiency_edge)
        + 0.035 * float(possession)
        + 0.18 * time_decay * float(score_diff)
    )
    return _clamp(0.001, _logit_prob(x), 0.999)


def monte_carlo_win_probability(
    state: LiveState,
    simulations: int = 2000,
    seed: Optional[int] = None,
) -> float:
    """
    Fast Monte Carlo repricing:
    - Derives possessions left from pace/time.
    - Simulates possession outcomes using a PPP-calibrated distribution.
    """
    sims = max(100, int(simulations))
    rng = random.Random(seed)
    remaining_possessions = max(1, int(round(state.pace * (max(0, state.time_remaining_sec) / 2880.0))))

    base_ppp = 1.10
    ppp_team = _clamp(0.75, base_ppp + (state.efficiency_edge / 2.0), 1.45)
    ppp_opp = _clamp(0.75, base_ppp - (state.efficiency_edge / 2.0), 1.45)

    wins = 0
    for _ in range(sims):
        diff = float(state.score_diff)
        poss_team = remaining_possessions // 2
        poss_opp = remaining_possessions // 2
        if state.possession > 0:
            poss_team += 1
        elif state.possession < 0:
            poss_opp += 1

        for _ in range(poss_team):
            diff += _sample_points(rng, ppp_team)
        for _ in range(poss_opp):
            diff -= _sample_points(rng, ppp_opp)

        if diff > 0:
            wins += 1
        elif diff == 0 and rng.random() < 0.5:
            wins += 1

    return _clamp(0.001, wins / sims, 0.999)
