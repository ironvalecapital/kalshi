from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random
import math


@dataclass
class StressScenario:
    name: str
    vol_mult: float
    funding_mult: float
    liq_spike: float
    depth_div: float
    spread_mult: float
    price_vel_mult: float


SCENARIOS = {
    "btc_panic": StressScenario("btc_panic", 2.5, 3.0, 2.8, 3.0, 1.8, 2.2),
    "btc_weekend_thin": StressScenario("btc_weekend_thin", 1.6, 1.8, 1.5, 2.2, 1.5, 1.4),
    "sports_panic": StressScenario("sports_panic", 2.0, 1.0, 1.6, 2.5, 1.7, 2.0),
}


@dataclass
class StressSummary:
    scenario: str
    mean_ev: float
    p05_pnl: float
    ruin_prob: float
    kelly_robustness: float
    stress_score: float
    risk_contraction: float


def _quantile(xs: List[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = int(max(0, min(len(ys) - 1, round(q * (len(ys) - 1)))))
    return ys[i]


def run_shadow_stress(
    scenario_key: str,
    base_edge: float = 0.06,
    base_kelly: float = 0.25,
    bankroll: float = 100.0,
    paths: int = 1000,
    trades_per_path: int = 150,
    seed: int = 7,
) -> StressSummary:
    s = SCENARIOS[scenario_key]
    rng = random.Random(seed)
    finals: List[float] = []
    ruins = 0
    evs: List[float] = []

    for _ in range(paths):
        eq = bankroll
        peak = bankroll
        ruined = False
        for _ in range(trades_per_path):
            edge = base_edge / max(1.0, s.spread_mult * 0.7 + s.vol_mult * 0.3)
            var = 0.03 * s.vol_mult + 0.01 * s.liq_spike
            p_win = max(0.05, min(0.95, 0.5 + edge - 0.02 * (s.price_vel_mult - 1.0)))
            kelly = base_kelly / max(1.0, s.vol_mult * 0.6 + s.liq_spike * 0.4)
            kelly = max(0.03, min(0.35, kelly))
            size = eq * min(0.05, kelly)
            pnl = size * (edge + rng.gauss(0.0, var))
            eq += pnl
            peak = max(peak, eq)
            if peak > 0 and (peak - eq) / peak >= 0.50:
                ruined = True
        finals.append(eq - bankroll)
        evs.append((eq - bankroll) / max(1.0, trades_per_path))
        if ruined:
            ruins += 1

    mean_ev = sum(evs) / len(evs)
    p05 = _quantile(finals, 0.05)
    ruin_prob = ruins / max(1, paths)

    # Kelly robustness: how much safe Kelly remains under stress.
    kelly_robust = max(0.1, min(1.0, 1.0 - ruin_prob * 2.5 - max(0.0, -p05) / max(1.0, bankroll)))
    stress_score = max(0.0, min(1.0, 0.5 * ruin_prob + 0.5 * max(0.0, -p05) / max(1.0, bankroll)))
    risk_contraction = max(0.2, min(1.0, 1.0 - stress_score))

    return StressSummary(
        scenario=s.name,
        mean_ev=float(mean_ev),
        p05_pnl=float(p05),
        ruin_prob=float(ruin_prob),
        kelly_robustness=float(kelly_robust),
        stress_score=float(stress_score),
        risk_contraction=float(risk_contraction),
    )
