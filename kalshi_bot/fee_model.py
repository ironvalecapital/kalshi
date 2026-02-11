from __future__ import annotations

import math


# Kalshi fee schedule: round_up(rate * C * P * (1-P)) in dollars.
# https://kalshi.com/docs/kalshi-fee-schedule.pdf


def fee_total_cents(count: int, price_cents: int, maker: bool = True) -> int:
    if count <= 0 or price_cents <= 0 or price_cents >= 100:
        return 0
    rate = 0.0175 if maker else 0.07
    p = price_cents / 100.0
    fee_dollars = rate * count * p * (1.0 - p)
    return int(math.ceil(fee_dollars * 100.0))


def fee_cents(count: int, price_cents: int, maker: bool = True) -> int:
    return fee_total_cents(count, price_cents, maker)


def fee_per_contract_cents(count: int, price_cents: int, maker: bool = True) -> float:
    total = fee_total_cents(count, price_cents, maker)
    return total / max(1, count)
