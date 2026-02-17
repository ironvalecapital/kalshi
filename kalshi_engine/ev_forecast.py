from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import math
import random


@dataclass
class TradeExpectation:
    win_prob: float
    price_prob: float
    size_usd: float
    fee_usd: float = 0.0


@dataclass
class EVForecastSummary:
    mean_pnl: float
    stdev_pnl: float
    p05: float
    p50: float
    p95: float
    cvar_05: float
    ruin_prob: float


def expected_trade_pnl(t: TradeExpectation) -> float:
    p = max(1e-6, min(1 - 1e-6, float(t.win_prob)))
    c = max(1e-6, min(1 - 1e-6, float(t.price_prob)))
    size = max(0.0, float(t.size_usd))
    fee = max(0.0, float(t.fee_usd))
    payout_mult = (1.0 - c) / c
    win_pnl = size * payout_mult - fee
    lose_pnl = -size - fee
    return p * win_pnl + (1.0 - p) * lose_pnl


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(max(0, min(len(ys) - 1, round(q * (len(ys) - 1)))))
    return float(ys[i])


def simulate_forward_distribution(
    trades: Iterable[TradeExpectation],
    n_paths: int = 2000,
    horizon_trades: int = 200,
    bankroll_start: float = 100.0,
    seed: int = 7,
) -> EVForecastSummary:
    universe = list(trades)
    if not universe:
        return EVForecastSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rng = random.Random(seed)
    finals: List[float] = []
    ruin = 0

    for _ in range(n_paths):
        eq = float(bankroll_start)
        peak = eq
        ruined = False
        for _ in range(horizon_trades):
            t = universe[rng.randrange(len(universe))]
            p = max(1e-6, min(1 - 1e-6, t.win_prob))
            c = max(1e-6, min(1 - 1e-6, t.price_prob))
            payout_mult = (1.0 - c) / c
            if rng.random() < p:
                pnl = t.size_usd * payout_mult - t.fee_usd
            else:
                pnl = -t.size_usd - t.fee_usd
            eq += pnl
            peak = max(peak, eq)
            if peak > 0 and (peak - eq) / peak >= 0.50:
                ruined = True
        finals.append(eq - bankroll_start)
        if ruined:
            ruin += 1

    mean = sum(finals) / len(finals)
    var = sum((x - mean) ** 2 for x in finals) / max(1, len(finals) - 1)
    stdev = math.sqrt(max(0.0, var))
    p05 = _quantile(finals, 0.05)
    p50 = _quantile(finals, 0.50)
    p95 = _quantile(finals, 0.95)
    tail = [x for x in finals if x <= p05] or [p05]
    cvar = sum(tail) / len(tail)

    return EVForecastSummary(
        mean_pnl=float(mean),
        stdev_pnl=float(stdev),
        p05=float(p05),
        p50=float(p50),
        p95=float(p95),
        cvar_05=float(cvar),
        ruin_prob=float(ruin / len(finals)),
    )


def bootstrap_pnl_distribution(
    historical_pnls: List[float],
    n_paths: int = 2000,
    horizon_trades: int = 200,
    seed: int = 7,
) -> EVForecastSummary:
    if not historical_pnls:
        return EVForecastSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rng = random.Random(seed)
    finals: List[float] = []
    ruin = 0

    for _ in range(n_paths):
        eq = 0.0
        peak = 0.0
        ruined = False
        for _ in range(horizon_trades):
            pnl = float(historical_pnls[rng.randrange(len(historical_pnls))])
            eq += pnl
            peak = max(peak, eq)
            base = max(1e-6, peak + 1.0)
            if (peak - eq) / base >= 0.50:
                ruined = True
        finals.append(eq)
        if ruined:
            ruin += 1

    mean = sum(finals) / len(finals)
    var = sum((x - mean) ** 2 for x in finals) / max(1, len(finals) - 1)
    stdev = math.sqrt(max(0.0, var))
    p05 = _quantile(finals, 0.05)
    p50 = _quantile(finals, 0.50)
    p95 = _quantile(finals, 0.95)
    tail = [x for x in finals if x <= p05] or [p05]
    cvar = sum(tail) / len(tail)

    return EVForecastSummary(
        mean_pnl=float(mean),
        stdev_pnl=float(stdev),
        p05=float(p05),
        p50=float(p50),
        p95=float(p95),
        cvar_05=float(cvar),
        ruin_prob=float(ruin / len(finals)),
    )


def forward_from_edge_distribution(
    edges: List[float],
    prices: List[float],
    size_usd: float,
    fee_usd: float = 0.0,
    n_paths: int = 2000,
    horizon_trades: int = 200,
    seed: int = 7,
) -> EVForecastSummary:
    """
    Build trade expectations from current edge distribution:
    p_model = clamp(price + edge), then simulate forward PnL distribution.
    """
    if not edges or not prices:
        return EVForecastSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    m = min(len(edges), len(prices))
    trades: List[TradeExpectation] = []
    for i in range(m):
        p_mkt = max(0.01, min(0.99, float(prices[i])))
        p_mod = max(0.01, min(0.99, p_mkt + float(edges[i])))
        trades.append(TradeExpectation(win_prob=p_mod, price_prob=p_mkt, size_usd=size_usd, fee_usd=fee_usd))
    return simulate_forward_distribution(
        trades=trades,
        n_paths=n_paths,
        horizon_trades=horizon_trades,
        bankroll_start=100.0,
        seed=seed,
    )
