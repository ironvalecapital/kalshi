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


def kelly_fraction_yes(
    p_yes: float,
    price_cents: int,
    fee_cents_per_contract: float = 0.0,
    slip_cents: float = 0.0,
) -> float:
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    p = max(0.0, min(1.0, p_yes))
    ev_net = 100.0 * p - price_cents - fee_cents_per_contract - slip_cents
    if ev_net <= 0:
        return 0.0
    denom = 100.0 - price_cents
    return max(0.0, min(1.0, ev_net / denom))


def kelly_fraction_no(
    p_yes: float,
    price_cents: int,
    fee_cents_per_contract: float = 0.0,
    slip_cents: float = 0.0,
) -> float:
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    p_no = 1.0 - max(0.0, min(1.0, p_yes))
    ev_net = 100.0 * p_no - price_cents - fee_cents_per_contract - slip_cents
    if ev_net <= 0:
        return 0.0
    denom = 100.0 - price_cents
    return max(0.0, min(1.0, ev_net / denom))


def kelly_contracts(
    bankroll_usd: float,
    price_cents: int,
    kelly_fraction: float,
    fractional: float,
    fill_prob: float | None = None,
    use_fill_prob: bool = False,
    max_contracts: int | None = None,
) -> int:
    if bankroll_usd <= 0 or price_cents <= 0:
        return 0
    if kelly_fraction <= 0 or fractional <= 0:
        return 0
    scale = fractional
    if use_fill_prob and fill_prob is not None:
        scale *= max(0.0, min(1.0, fill_prob))
    dollars = bankroll_usd * kelly_fraction * scale
    cost_per = price_cents / 100.0
    if cost_per <= 0:
        return 0
    contracts = int(dollars / cost_per)
    if max_contracts is not None:
        contracts = min(contracts, max_contracts)
    return max(0, contracts)
