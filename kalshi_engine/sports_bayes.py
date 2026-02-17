from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class SportsGameState:
    score_diff: float  # team_of_interest - opponent
    time_remaining_min: float
    has_possession: bool
    timeouts_team: int
    timeouts_opp: int
    pregame_prob: float
    historical_scoring_rate_per_min: float
    current_kalshi_implied: float
    pace_factor: float = 1.0
    weather_factor: float = 1.0


@dataclass
class SportsBayesOutput:
    model_prob: float
    confidence_low: float
    confidence_high: float
    scoring_volatility_estimate: float
    time_decay_factor: float
    edge: float


def _clip_prob(p: float) -> float:
    return max(1e-6, min(1 - 1e-6, float(p)))


def _logit(p: float) -> float:
    p = _clip_prob(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def simulate_game(state: SportsGameState, n: int = 50_000, seed: int = 7) -> np.ndarray:
    """
    Monte Carlo over remaining scoring differential.
    """
    rng = np.random.default_rng(seed)
    t = max(0.0, float(state.time_remaining_min))
    base_rate = max(0.01, float(state.historical_scoring_rate_per_min))
    rate = base_rate * max(0.3, state.pace_factor) * max(0.3, state.weather_factor)
    # crude possession/timeout adjustments
    pos_adj = 0.35 if state.has_possession else -0.20
    to_adj = 0.08 * (state.timeouts_team - state.timeouts_opp)

    # Remaining points for both teams from Poisson approximation
    lam_team = max(0.05, (rate + pos_adj + to_adj) * t)
    lam_opp = max(0.05, (rate - pos_adj - to_adj) * t)
    team_pts = rng.poisson(lam=lam_team, size=n)
    opp_pts = rng.poisson(lam=lam_opp, size=n)
    final_diff = state.score_diff + (team_pts - opp_pts)
    return final_diff


def bayesian_win_probability(state: SportsGameState, n_sims: int = 50_000) -> SportsBayesOutput:
    sims = simulate_game(state, n=n_sims)
    live_prob = float(np.mean(sims > 0))
    vol = float(np.std(sims))

    # Dynamic weights: early game relies on prior; late game relies on live simulation.
    time_frac = max(0.0, min(1.0, state.time_remaining_min / 48.0))
    prior_w = 0.70 * time_frac + 0.10
    live_w = 1.0 - prior_w

    post_logit = prior_w * _logit(state.pregame_prob) + live_w * _logit(live_prob)
    post = _clip_prob(_sigmoid(post_logit))

    # Confidence band from MC quantiles
    wins = (sims > 0).astype(float)
    # Wilson-like approximation
    p = np.mean(wins)
    se = math.sqrt(max(1e-9, p * (1 - p) / len(wins)))
    c_low = _clip_prob(p - 1.96 * se)
    c_high = _clip_prob(p + 1.96 * se)
    edge = post - _clip_prob(state.current_kalshi_implied)
    return SportsBayesOutput(
        model_prob=post,
        confidence_low=c_low,
        confidence_high=c_high,
        scoring_volatility_estimate=vol,
        time_decay_factor=(1.0 - time_frac),
        edge=edge,
    )
