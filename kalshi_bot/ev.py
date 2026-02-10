from __future__ import annotations

import math


def estimate_fee_cents(count: int, price_cents: int, maker: bool, maker_rate: float, taker_rate: float) -> int:
    # Kalshi fee schedule: round_up(rate * C * P * (1-P)) in dollars.
    if count <= 0 or price_cents <= 0 or price_cents >= 100:
        return 0
    rate = maker_rate if maker else taker_rate
    p = price_cents / 100.0
    fee = rate * count * p * (1.0 - p)
    return int(math.ceil(fee * 100.0))


def ev_buy_yes_cents(p_yes: float, price_cents: int) -> float:
    p = max(0.0, min(1.0, p_yes))
    price = price_cents / 100.0
    return (p * (1.0 - price) - (1.0 - p) * price) * 100.0


def ev_buy_no_cents(p_yes: float, price_cents: int) -> float:
    p = max(0.0, min(1.0, p_yes))
    price = price_cents / 100.0
    return ((1.0 - p) * (1.0 - price) - p * price) * 100.0


def fill_probability(
    spread_cents: float,
    trades_1h: int,
    time_to_close_hours: float,
    depth: int = 0,
    trades_weight: float = 0.02,
    depth_weight: float = 0.005,
) -> float:
    if spread_cents <= 0:
        spread_cents = 1.0
    base = min(0.95, 0.12 + trades_weight * trades_1h + depth_weight * depth)
    time_factor = 0.5 if time_to_close_hours > 24 else 1.0
    spread_factor = max(0.1, 1.0 - (spread_cents / 20.0))
    return max(0.0, min(0.99, base * time_factor * spread_factor))


def spread_penalty_cents(spread_cents: float) -> float:
    return max(0.0, spread_cents * 0.1)
