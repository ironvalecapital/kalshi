from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import math

from .ev_forecast import EVForecastSummary, bootstrap_pnl_distribution


@dataclass
class EdgeStats:
    mean: float
    stdev: float
    win_rate: float
    skew: float
    kurtosis: float


@dataclass
class CapitalHealth:
    expected_return_30d: float
    worst_5pct_drawdown: float
    negative_month_prob: float
    risk_of_ruin: float
    kelly_safety_factor: float
    ev_ratio: float
    sharpe_like: float


def _moment_stats(xs: List[float]) -> EdgeStats:
    if not xs:
        return EdgeStats(0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / max(1, n - 1)
    sd = math.sqrt(max(1e-12, var))
    m3 = sum((x - mu) ** 3 for x in xs) / max(1, n)
    m4 = sum((x - mu) ** 4 for x in xs) / max(1, n)
    skew = m3 / (sd ** 3 + 1e-12)
    kurt = m4 / (sd ** 4 + 1e-12) - 3.0
    win = sum(1 for x in xs if x > 0) / n
    return EdgeStats(mean=mu, stdev=sd, win_rate=win, skew=skew, kurtosis=kurt)


def recommend_kelly_safety(
    ev_ratio: float,
    ruin_prob: float,
    worst_5pct_drawdown: float,
) -> float:
    # Base safety from EV delivery.
    s = max(0.35, min(1.0, float(ev_ratio)))
    # Penalize ruin + drawdown risk.
    s *= max(0.35, 1.0 - 2.0 * float(ruin_prob))
    if worst_5pct_drawdown < -0.20:
        s *= 0.80
    if worst_5pct_drawdown < -0.35:
        s *= 0.70
    return max(0.20, min(1.0, s))


def build_capital_health_report(
    trade_pnls: List[float],
    expected_edges: Optional[List[float]] = None,
    n_paths: int = 10000,
    horizon_trades: int = 300,
) -> Dict[str, Any]:
    expected_edges = expected_edges or []
    edge_stats = _moment_stats(expected_edges)
    pnl_stats = _moment_stats(trade_pnls)

    forecast: EVForecastSummary = bootstrap_pnl_distribution(
        historical_pnls=trade_pnls,
        n_paths=n_paths,
        horizon_trades=horizon_trades,
        seed=7,
    )

    realized_ev = sum(trade_pnls)
    expected_ev = sum(expected_edges) if expected_edges else 0.0
    ev_ratio = (realized_ev / expected_ev) if abs(expected_ev) > 1e-9 else 1.0
    sharpe_like = pnl_stats.mean / max(1e-9, pnl_stats.stdev)

    health = CapitalHealth(
        expected_return_30d=float(forecast.mean_pnl),
        worst_5pct_drawdown=float(forecast.p05),
        negative_month_prob=float(0.0 if not trade_pnls else sum(1 for x in trade_pnls if x < 0) / len(trade_pnls)),
        risk_of_ruin=float(forecast.ruin_prob),
        kelly_safety_factor=float(recommend_kelly_safety(ev_ratio=ev_ratio, ruin_prob=forecast.ruin_prob, worst_5pct_drawdown=forecast.p05)),
        ev_ratio=float(ev_ratio),
        sharpe_like=float(sharpe_like),
    )

    stress_level = "GREEN"
    if health.risk_of_ruin >= 0.20 or health.worst_5pct_drawdown <= -0.35:
        stress_level = "RED"
    elif health.risk_of_ruin >= 0.10 or health.worst_5pct_drawdown <= -0.20:
        stress_level = "YELLOW"

    return {
        "edge_stats": edge_stats.__dict__,
        "pnl_stats": pnl_stats.__dict__,
        "forecast": forecast.__dict__,
        "capital_health": health.__dict__,
        "stress_level": stress_level,
        "global_kelly_recommendation": 0.25 * health.kelly_safety_factor,
    }
