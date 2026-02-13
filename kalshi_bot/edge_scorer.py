from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ev import estimate_fee_cents, ev_buy_no_cents, ev_buy_yes_cents


@dataclass
class EdgeScore:
    ticker: str
    side: str
    price_cents: int
    p_model: float
    p_implied: float
    spread_cents: float
    fee_total_cents: int
    fee_per_contract_cents: float
    slippage_cents: float
    ev_cents: float
    edge_cents: float


def implied_probability(price_cents: Optional[int]) -> Optional[float]:
    if price_cents is None:
        return None
    if price_cents <= 0 or price_cents >= 100:
        return None
    return price_cents / 100.0


def score_yes(
    ticker: str,
    price_cents: int,
    p_model: float,
    spread_cents: float,
    count: int,
    maker: bool,
    maker_rate: float,
    taker_rate: float,
    slippage_cents: float = 0.0,
) -> EdgeScore:
    p_implied = implied_probability(price_cents) or 0.0
    fee_total = estimate_fee_cents(count, price_cents, maker, maker_rate, taker_rate)
    fee_per = fee_total / max(1, count)
    ev_gross = ev_buy_yes_cents(p_model, price_cents)
    ev_net = ev_gross - fee_per - slippage_cents - spread_cents
    edge_cents = (p_model - p_implied) * 100.0
    return EdgeScore(
        ticker=ticker,
        side="yes",
        price_cents=price_cents,
        p_model=p_model,
        p_implied=p_implied,
        spread_cents=spread_cents,
        fee_total_cents=fee_total,
        fee_per_contract_cents=fee_per,
        slippage_cents=slippage_cents,
        ev_cents=ev_net,
        edge_cents=edge_cents,
    )


def score_no(
    ticker: str,
    price_cents: int,
    p_model: float,
    spread_cents: float,
    count: int,
    maker: bool,
    maker_rate: float,
    taker_rate: float,
    slippage_cents: float = 0.0,
) -> EdgeScore:
    p_implied = implied_probability(price_cents) or 0.0
    fee_total = estimate_fee_cents(count, price_cents, maker, maker_rate, taker_rate)
    fee_per = fee_total / max(1, count)
    ev_gross = ev_buy_no_cents(p_model, price_cents)
    ev_net = ev_gross - fee_per - slippage_cents - spread_cents
    edge_cents = ((1.0 - p_model) - p_implied) * 100.0
    return EdgeScore(
        ticker=ticker,
        side="no",
        price_cents=price_cents,
        p_model=p_model,
        p_implied=p_implied,
        spread_cents=spread_cents,
        fee_total_cents=fee_total,
        fee_per_contract_cents=fee_per,
        slippage_cents=slippage_cents,
        ev_cents=ev_net,
        edge_cents=edge_cents,
    )
