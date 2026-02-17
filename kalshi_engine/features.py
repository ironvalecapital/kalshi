from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterable, Optional

import numpy as np

from .models import LLMSnapshot, RiskFlags


def _best_bid(levels: list[tuple[int, int]]) -> Optional[int]:
    return levels[0][0] if levels else None


def _sum_depth(levels: list[tuple[int, int]], k: int = 5) -> int:
    return int(sum(s for _, s in levels[:k])) if levels else 0


def implied_mid_prob(best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> float:
    if best_yes_bid is not None and best_no_bid is not None:
        yes_ask = 100 - best_no_bid
        return max(0.0, min(1.0, (best_yes_bid + yes_ask) / 200.0))
    if best_yes_bid is not None:
        return max(0.0, min(1.0, best_yes_bid / 100.0))
    if best_no_bid is not None:
        return max(0.0, min(1.0, 1.0 - best_no_bid / 100.0))
    return 0.5


def spread_cents(best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> Optional[int]:
    if best_yes_bid is None or best_no_bid is None:
        return None
    yes_ask = 100 - best_no_bid
    return int(yes_ask - best_yes_bid)


def imbalance(depth_yes: int, depth_no: int) -> float:
    d = depth_yes + depth_no
    return 0.0 if d <= 0 else (depth_yes - depth_no) / d


def volume_velocity(trades: Iterable[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, float]:
    now = now or datetime.now(timezone.utc)
    windows = {"1m": 60, "5m": 300, "30m": 1800}
    out: Dict[str, float] = {}
    rows = list(trades)
    for name, sec in windows.items():
        total = 0.0
        for t in rows:
            ts = _to_dt(t.get("ts") or t.get("timestamp"))
            if ts is None:
                continue
            if (now - ts).total_seconds() <= sec:
                total += float(t.get("count") or t.get("size") or 0)
        out[name] = total / max(1.0, sec)
    return out


def price_velocity(trades: Iterable[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, float]:
    now = now or datetime.now(timezone.utc)
    windows = {"1m": 60, "5m": 300, "30m": 1800}
    rows = list(trades)
    out: Dict[str, float] = {}
    for name, sec in windows.items():
        pts = []
        for t in rows:
            ts = _to_dt(t.get("ts") or t.get("timestamp"))
            p = t.get("yes_price") or t.get("price")
            if ts is None or p is None:
                continue
            if (now - ts).total_seconds() <= sec:
                pts.append((ts, float(p) / 100.0))
        pts.sort(key=lambda x: x[0])
        if len(pts) < 2:
            out[name] = 0.0
        else:
            dt = max(1.0, (pts[-1][0] - pts[0][0]).total_seconds())
            out[name] = (pts[-1][1] - pts[0][1]) / dt
    return out


def estimate_slippage(target_size: int, levels: list[tuple[int, int]]) -> float:
    """
    Estimated slippage in cents from walking book by target contracts.
    """
    if target_size <= 0 or not levels:
        return 0.0
    need = int(target_size)
    top_price = levels[0][0]
    paid = 0
    notional = 0
    for p, s in levels:
        take = min(need, int(s))
        notional += take * int(p)
        paid += take
        need -= take
        if need <= 0:
            break
    if paid <= 0:
        return 0.0
    avg = notional / paid
    return float(abs(avg - top_price))


def short_term_reversion_signal(trades: Iterable[Dict[str, Any]], current_price_prob: float, lookback_sec: int = 120) -> Dict[str, float]:
    now = datetime.now(timezone.utc)
    pts = []
    for t in trades:
        ts = _to_dt(t.get("ts") or t.get("timestamp"))
        p = t.get("yes_price") or t.get("price")
        if ts is None or p is None:
            continue
        if (now - ts).total_seconds() <= lookback_sec:
            pts.append(float(p) / 100.0)
    if len(pts) < 4:
        return {"vwap_2m": current_price_prob, "deviation": 0.0, "fade_signal": 0.0}
    arr = np.asarray(pts, dtype=float)
    vwap = float(np.mean(arr))
    deviation = float(current_price_prob - vwap)
    fade = float(-deviation)
    return {"vwap_2m": vwap, "deviation": deviation, "fade_signal": fade}


def emotion_spike_score(
    price_vel_1m: float,
    baseline_vol: float,
    volume_spike_ratio: float,
    spread_widening_ratio: float,
    depth_thinning_ratio: float,
) -> float:
    z = abs(price_vel_1m) / max(1e-6, baseline_vol)
    ess = z * 0.4 + volume_spike_ratio * 0.3 + spread_widening_ratio * 0.2 + depth_thinning_ratio * 0.1
    return float(max(0.0, ess))


def btc_tilt_score(
    funding_extreme_z: float,
    liquidation_spike: float,
    orderbook_imbalance_ratio: float,
    weekend_thin_liquidity_flag: float,
) -> float:
    score = (
        funding_extreme_z * 0.3
        + liquidation_spike * 0.3
        + orderbook_imbalance_ratio * 0.2
        + weekend_thin_liquidity_flag * 0.2
    )
    return float(max(0.0, score))


def build_llm_market_snapshot(
    market_ticker: str,
    event_ticker: Optional[str],
    close_time: Optional[str],
    status: Optional[str],
    yes_levels: list[tuple[int, int]],
    no_levels: list[tuple[int, int]],
    trades: Iterable[Dict[str, Any]],
    model_prob: float,
    bankroll: float,
    target_order_size: int,
    drawdown_regime: bool = False,
    exposure_cap_hit: bool = False,
    min_edge_threshold: float = 0.06,
    mode: str = "GTO",
) -> LLMSnapshot:
    by = _best_bid(yes_levels)
    bn = _best_bid(no_levels)
    sp = spread_cents(by, bn)
    implied = implied_mid_prob(by, bn)
    dy = _sum_depth(yes_levels, 5)
    dn = _sum_depth(no_levels, 5)
    imb = imbalance(dy, dn)
    vv = volume_velocity(trades)
    pv = price_velocity(trades)
    slip = estimate_slippage(target_order_size, yes_levels)
    rev = short_term_reversion_signal(trades, current_price_prob=implied)

    edge = float(model_prob - implied)
    risk_flags = RiskFlags(
        drawdown_regime=drawdown_regime,
        exposure_cap_hit=exposure_cap_hit,
        thin_book=(dy + dn) < max(10, target_order_size * 3),
        no_trade_reason=None if edge > min_edge_threshold else "edge_below_threshold",
    )
    return LLMSnapshot(
        market_ticker=market_ticker,
        context={
            "event_ticker": event_ticker,
            "close_time": close_time,
            "status": status,
            "mode": mode,
        },
        liquidity={
            "implied_prob": implied,
            "best_yes_bid": by,
            "best_no_bid": bn,
            "spread": sp,
            "depth_yes_5lvls": dy,
            "depth_no_5lvls": dn,
            "imbalance": imb,
            "slippage_estimate": slip,
        },
        flow={
            "volume_velocity_1m": vv["1m"],
            "volume_velocity_5m": vv["5m"],
            "volume_velocity_30m": vv["30m"],
            "price_velocity_1m": pv["1m"],
            "price_velocity_5m": pv["5m"],
            "price_velocity_30m": pv["30m"],
            "short_term_mean": rev["vwap_2m"],
            "short_term_deviation": rev["deviation"],
            "short_term_fade_signal": rev["fade_signal"],
        },
        model={
            "model_prob": float(model_prob),
            "edge": edge,
            "min_edge_threshold": min_edge_threshold,
            "bankroll": bankroll,
            "target_order_size": target_order_size,
        },
        risk=risk_flags.model_dump(),
    )


def _to_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    try:
        s = str(v).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
