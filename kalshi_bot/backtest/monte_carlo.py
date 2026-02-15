from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MonteCarloConfig:
    bankroll_start: float = 5.0
    trades_per_year: int = 500
    years: int = 5
    paths: int = 10_000
    win_prob: float = 0.57
    # Net edge model as multiplicative return per trade:
    # win -> +edge_up * stake, loss -> -edge_down * stake
    edge_up: float = 0.05
    edge_down: float = 1.0
    kelly_fraction: float = 0.20
    max_per_trade: float = 0.05
    drawdown_cut_trigger: float = 0.20
    drawdown_cut_mult: float = 0.5
    seed: int = 7


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    i = max(0, min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1))))
    return float(sorted_values[i])


def run_growth_simulation(cfg: MonteCarloConfig) -> Dict[str, float]:
    rng = random.Random(cfg.seed)
    n_trades = cfg.trades_per_year * cfg.years
    finals: List[float] = []
    max_dds: List[float] = []
    ruin_count = 0
    deep_dd_count = 0

    for _ in range(cfg.paths):
        bankroll = cfg.bankroll_start
        peak = bankroll
        max_dd = 0.0

        for _ in range(n_trades):
            if bankroll <= 0.01:
                ruin_count += 1
                break
            dd = 0.0 if peak <= 0 else (peak - bankroll) / peak
            k_mult = cfg.drawdown_cut_mult if dd >= cfg.drawdown_cut_trigger else 1.0
            f = min(cfg.max_per_trade, cfg.kelly_fraction * k_mult)
            stake = bankroll * max(0.0, f)

            if rng.random() < cfg.win_prob:
                bankroll += stake * cfg.edge_up
            else:
                bankroll -= stake * cfg.edge_down
            peak = max(peak, bankroll)
            if peak > 0:
                max_dd = max(max_dd, (peak - bankroll) / peak)

        finals.append(bankroll)
        max_dds.append(max_dd)
        if max_dd >= 0.70:
            deep_dd_count += 1

    finals_sorted = sorted(finals)
    dd_sorted = sorted(max_dds)
    return {
        "median_final": _quantile(finals_sorted, 0.50),
        "p25_final": _quantile(finals_sorted, 0.25),
        "p75_final": _quantile(finals_sorted, 0.75),
        "p95_final": _quantile(finals_sorted, 0.95),
        "median_max_dd": _quantile(dd_sorted, 0.50),
        "ruin_prob": ruin_count / max(1, cfg.paths),
        "deep_dd_prob": deep_dd_count / max(1, cfg.paths),
    }


def style_configs() -> Dict[str, MonteCarloConfig]:
    return {
        "aggressive": MonteCarloConfig(kelly_fraction=0.50, max_per_trade=0.10),
        "institutional": MonteCarloConfig(kelly_fraction=0.12, max_per_trade=0.04),
        "adaptive": MonteCarloConfig(kelly_fraction=0.25, max_per_trade=0.05, drawdown_cut_mult=0.5),
    }

