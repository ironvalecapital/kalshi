from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DecisionState:
    ev_yes_cents: float
    ev_no_cents: float
    spread_cents: float
    volatility: float
    liquidity: float
    time_to_close_hours: float
    risk_penalty: float


def eval_state(state: DecisionState, action: str, weights: Dict[str, float]) -> float:
    ev = state.ev_yes_cents if action == "buy_yes" else state.ev_no_cents if action == "buy_no" else 0.0
    return (
        weights.get("w_ev", 1.0) * ev
        - weights.get("w_spread", 0.5) * state.spread_cents
        - weights.get("w_vol", 0.5) * state.volatility
        + weights.get("w_liq", 0.2) * state.liquidity
        - weights.get("w_risk", 1.0) * state.risk_penalty
    )


def adversary_response(state: DecisionState, action: str) -> List[DecisionState]:
    responses = []
    for spread_bump in (0.0, 2.0, 4.0):
        responses.append(
            DecisionState(
                ev_yes_cents=state.ev_yes_cents - spread_bump * 0.5,
                ev_no_cents=state.ev_no_cents - spread_bump * 0.5,
                spread_cents=state.spread_cents + spread_bump,
                volatility=state.volatility + spread_bump * 0.1,
                liquidity=state.liquidity,
                time_to_close_hours=state.time_to_close_hours,
                risk_penalty=state.risk_penalty,
            )
        )
    return responses


def alpha_beta(
    state: DecisionState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    weights: Dict[str, float],
    actions: List[str],
) -> Tuple[float, str]:
    if depth == 0 or not actions:
        return max(eval_state(state, a, weights) for a in actions) if actions else 0.0, "hold"

    if maximizing:
        best_val = float("-inf")
        best_action = "hold"
        for action in actions:
            next_states = adversary_response(state, action)
            value = float("inf")
            for s in next_states:
                v, _ = alpha_beta(s, depth - 1, alpha, beta, False, weights, actions)
                value = min(value, v)
            if value > best_val:
                best_val = value
                best_action = action
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return best_val, best_action

    best_val = float("inf")
    best_action = "hold"
    for action in actions:
        v = eval_state(state, action, weights)
        if v < best_val:
            best_val = v
            best_action = action
        beta = min(beta, v)
        if beta <= alpha:
            break
    return best_val, best_action


def choose_action(state: DecisionState, weights: Dict[str, float], depth: int = 2) -> Tuple[str, float]:
    actions = ["buy_yes", "buy_no", "hold"]
    score, action = alpha_beta(state, depth, float("-inf"), float("inf"), True, weights, actions)
    return action, score
