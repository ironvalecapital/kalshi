from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple
import math


@dataclass
class TradeRecord:
    ts: datetime
    pnl: float
    regime: str
    strategy_type: str


def _sharpe(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    sd = math.sqrt(max(1e-12, var))
    return mu / sd


def trade_level_sharpe(pnls: List[float]) -> float:
    return _sharpe(pnls)


def daily_sharpe(records: Iterable[TradeRecord]) -> float:
    by_day: Dict[str, float] = {}
    for r in records:
        day = r.ts.date().isoformat()
        by_day[day] = by_day.get(day, 0.0) + float(r.pnl)
    return _sharpe(list(by_day.values()))


def annualized_from_daily(sharpe_daily: float) -> float:
    return float(sharpe_daily) * math.sqrt(252.0)


def regime_adjusted_sharpe(records: Iterable[TradeRecord]) -> Dict[str, float]:
    by_regime: Dict[str, List[float]] = {}
    for r in records:
        by_regime.setdefault(r.regime, []).append(float(r.pnl))
    return {k: _sharpe(v) for k, v in by_regime.items()}


def strategy_type_sharpe(records: Iterable[TradeRecord]) -> Dict[str, float]:
    by_type: Dict[str, List[float]] = {}
    for r in records:
        by_type.setdefault(r.strategy_type, []).append(float(r.pnl))
    return {k: _sharpe(v) for k, v in by_type.items()}


def allocation_weights_from_sharpe(sharpe_by_type: Dict[str, float], min_weight: float = 0.05) -> Dict[str, float]:
    positive = {k: max(0.0, v) for k, v in sharpe_by_type.items()}
    s = sum(positive.values())
    if s <= 0:
        n = max(1, len(sharpe_by_type))
        return {k: 1.0 / n for k in sharpe_by_type}
    raw = {k: v / s for k, v in positive.items()}
    # floor low sharpe to tiny weight, then renormalize
    floored = {k: max(min_weight if positive[k] <= 0.1 else 0.0, w) for k, w in raw.items()}
    z = sum(floored.values())
    return {k: v / z for k, v in floored.items()}
