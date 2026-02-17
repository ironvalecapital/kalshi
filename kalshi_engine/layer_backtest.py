from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math
import random


@dataclass
class LayerResult:
    name: str
    trades: int
    win_rate: float
    avg_edge: float
    pnl: float
    max_drawdown: float


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak > 0:
            max_dd = max(max_dd, (peak - x) / peak)
    return max_dd


def _kelly_fraction_binary(p: float, price: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    q = 1.0 - p
    c = max(1e-6, min(1 - 1e-6, price))
    b = (1.0 - c) / c
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))


def run_layer_comparison(seed: int = 7, steps: int = 2500, bankroll_start: float = 100.0) -> List[LayerResult]:
    rng = random.Random(seed)
    layers = [
        "pure_prob",
        "prob_plus_exploit",
        "prob_exploit_micro",
        "full_internal_arb",
    ]
    results: Dict[str, Dict[str, float | int | list[float]]] = {}
    for layer in layers:
        results[layer] = {
            "bankroll": bankroll_start,
            "wins": 0,
            "trades": 0,
            "edge_sum": 0.0,
            "equity": [bankroll_start],
        }

    for i in range(steps):
        # Simulated market context
        regime = "calm"
        r = rng.random()
        if r < 0.15:
            regime = "panic"
        elif r < 0.45:
            regime = "expansion"

        is_weekend_btc = (i % 7) in (5, 6)
        thin_liquidity = rng.random() < (0.45 if regime == "panic" else 0.20)
        sports_spike = rng.random() < (0.30 if regime == "expansion" else 0.08)

        implied = min(0.95, max(0.05, 0.5 + rng.uniform(-0.18, 0.18)))
        model = min(0.97, max(0.03, implied + rng.uniform(-0.10, 0.12)))
        edge = model - implied

        # Outcome probability: slight true edge if model diverges correctly.
        p_true = min(0.95, max(0.05, implied + 0.35 * edge + rng.uniform(-0.02, 0.02)))

        for layer in layers:
            s = results[layer]
            bankroll = float(s["bankroll"])
            if bankroll <= 0:
                continue

            edge_thresh = 0.06
            kelly_mult = 0.25
            spread_cost = 0.003

            if layer in {"prob_plus_exploit", "prob_exploit_micro", "full_internal_arb"}:
                if regime == "panic" or sports_spike or (is_weekend_btc and thin_liquidity):
                    edge_thresh *= 0.90
                    kelly_mult *= 1.15

            if layer in {"prob_exploit_micro", "full_internal_arb"}:
                # microstructure filter
                if thin_liquidity:
                    edge_thresh *= 1.10
                spread_cost *= 0.70

            if layer == "full_internal_arb":
                # internal arb provides extra opportunities in stressed books
                if thin_liquidity and rng.random() < 0.20:
                    edge += 0.02
                if regime == "panic" and rng.random() < 0.12:
                    edge += 0.01

            if edge < edge_thresh:
                s["equity"].append(bankroll)
                continue

            f_star = _kelly_fraction_binary(model, implied)
            f = min(0.02, max(0.0, f_star * kelly_mult))
            size = bankroll * f
            if size <= 0:
                s["equity"].append(bankroll)
                continue

            s["trades"] = int(s["trades"]) + 1
            s["edge_sum"] = float(s["edge_sum"]) + edge

            win = rng.random() < p_true
            if win:
                pnl = size * ((1.0 - implied) / max(implied, 1e-6))
                s["wins"] = int(s["wins"]) + 1
            else:
                pnl = -size
            pnl -= size * spread_cost

            bankroll = bankroll + pnl
            s["bankroll"] = bankroll
            s["equity"].append(bankroll)

    out: List[LayerResult] = []
    for layer in layers:
        s = results[layer]
        trades = int(s["trades"])
        wins = int(s["wins"])
        edge_sum = float(s["edge_sum"])
        equity = list(s["equity"])
        out.append(
            LayerResult(
                name=layer,
                trades=trades,
                win_rate=(wins / trades) if trades > 0 else 0.0,
                avg_edge=(edge_sum / trades) if trades > 0 else 0.0,
                pnl=float(s["bankroll"]) - bankroll_start,
                max_drawdown=_max_drawdown(equity),
            )
        )
    return out
