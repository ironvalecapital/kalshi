from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math


@dataclass
class CapacityPoint:
    capital: float
    order_size: float
    slippage: float
    effective_edge: float
    sharpe_like: float


@dataclass
class CapacityScanResult:
    points: List[CapacityPoint]
    recommended_max_capital: float


def estimate_max_size(avg_depth_top5: float, liquidity_factor: float = 0.4) -> float:
    return max(0.0, float(avg_depth_top5) * float(liquidity_factor))


def slippage_estimate(order_size: float, total_depth: float) -> float:
    if total_depth <= 0:
        return 1.0
    return max(0.0, float(order_size) / float(total_depth))


def scan_sharpe_vs_capital(
    capitals: List[float],
    avg_depth_top5: float,
    avg_spread: float,
    volatility: float,
    edge: float,
) -> CapacityScanResult:
    pts: List[CapacityPoint] = []
    max_size = estimate_max_size(avg_depth_top5, liquidity_factor=0.4)
    for c in capitals:
        order_size = min(max_size, c * 0.02)
        slip = slippage_estimate(order_size, avg_depth_top5)
        spread_cost = max(0.0, avg_spread / 100.0)
        effective_edge = float(edge) - (slip + spread_cost)
        sharpe_like = effective_edge / max(1e-6, float(volatility))
        pts.append(
            CapacityPoint(
                capital=float(c),
                order_size=float(order_size),
                slippage=float(slip),
                effective_edge=float(effective_edge),
                sharpe_like=float(sharpe_like),
            )
        )

    valid = [p for p in pts if p.effective_edge > 0]
    if not valid:
        rec = float(min(capitals) if capitals else 0.0)
    else:
        rec = max(valid, key=lambda p: p.sharpe_like).capital
    return CapacityScanResult(points=pts, recommended_max_capital=rec)
