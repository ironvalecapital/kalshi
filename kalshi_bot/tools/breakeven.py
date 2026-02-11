from __future__ import annotations

from typing import Dict

from ..fee_model import fee_total_cents, fee_per_contract_cents


# Orderbook bids-only; asks derived via complement.
# https://docs.kalshi.com/api-reference/markets/get-market-orderbook


def breakeven_yes(price_cents: int, count: int, maker: bool, slip_cents: float) -> Dict[str, float]:
    fee_total = fee_total_cents(count, price_cents, maker)
    fee_per = fee_per_contract_cents(count, price_cents, maker)
    p_mkt = price_cents / 100.0
    p_star = (price_cents + fee_per + slip_cents) / 100.0
    delta_p = p_star - p_mkt
    return {
        "price_cents": price_cents,
        "fee_total_cents": fee_total,
        "fee_per_contract_cents": fee_per,
        "slip_cents": slip_cents,
        "p_market": p_mkt,
        "p_break_even": p_star,
        "delta_p": delta_p,
        "edge_cents": delta_p * 100.0,
    }


def breakeven_no(price_cents: int, count: int, maker: bool, slip_cents: float) -> Dict[str, float]:
    fee_total = fee_total_cents(count, price_cents, maker)
    fee_per = fee_per_contract_cents(count, price_cents, maker)
    p_no_mkt = price_cents / 100.0
    p_no_star = (price_cents + fee_per + slip_cents) / 100.0
    p_yes_star = 1.0 - p_no_star
    return {
        "price_cents": price_cents,
        "fee_total_cents": fee_total,
        "fee_per_contract_cents": fee_per,
        "slip_cents": slip_cents,
        "p_market_no": p_no_mkt,
        "p_break_even_no": p_no_star,
        "p_break_even_yes": p_yes_star,
        "delta_p": p_no_star - p_no_mkt,
        "edge_cents": (p_no_star - p_no_mkt) * 100.0,
    }
