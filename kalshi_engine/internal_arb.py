from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ArbOpportunity:
    market_ticker: str
    side: str
    edge_cents: float
    reason: str


@dataclass
class EventBasketOpportunity:
    event_ticker: str
    total_implied_prob: float
    gap: float
    action: str
    reason: str


def complementary_arb_edge(yes_ask_cents: Optional[int], no_ask_cents: Optional[int]) -> float:
    """
    If yes_ask + no_ask < 100, a positive synthetic edge may exist.
    """
    if yes_ask_cents is None or no_ask_cents is None:
        return 0.0
    return float(100 - (yes_ask_cents + no_ask_cents))


def event_consistency_edges(market_probs: Dict[str, float], target_sum: float = 1.0) -> Dict[str, float]:
    """
    For mutually exclusive outcomes under one event:
    edge_i = target_sum - sum(probs).
    Positive means underpriced basket, negative means overpriced.
    """
    s = float(sum(max(0.0, min(1.0, p)) for p in market_probs.values()))
    gap = target_sum - s
    return {k: gap for k in market_probs.keys()}


def event_basket_opportunity(
    event_ticker: str,
    market_probs: Dict[str, float],
    target_sum: float = 1.0,
    min_gap: float = 0.03,
) -> Optional[EventBasketOpportunity]:
    """
    Internal Kalshi-only structural check:
    If mutually exclusive market probabilities in one event don't sum to ~1,
    basket opportunity may exist.
    """
    s = float(sum(max(0.0, min(1.0, p)) for p in market_probs.values()))
    gap = target_sum - s
    if abs(gap) < min_gap:
        return None
    action = "buy_underpriced_basket" if gap > 0 else "sell_overpriced_basket"
    return EventBasketOpportunity(
        event_ticker=event_ticker,
        total_implied_prob=s,
        gap=gap,
        action=action,
        reason="event_sum_probability_mispricing",
    )


def find_internal_kalshi_opportunities(
    market_ticker: str,
    yes_bid: Optional[int],
    no_bid: Optional[int],
    yes_ask: Optional[int],
    no_ask: Optional[int],
    model_prob: float,
    edge_threshold_cents: float = 2.0,
) -> List[ArbOpportunity]:
    out: List[ArbOpportunity] = []
    if yes_ask is not None:
        edge_yes = (model_prob - yes_ask / 100.0) * 100.0
        if edge_yes >= edge_threshold_cents:
            out.append(ArbOpportunity(market_ticker, "buy_yes", edge_yes, "model_vs_yes_ask"))
    if no_ask is not None:
        edge_no = ((1.0 - model_prob) - no_ask / 100.0) * 100.0
        if edge_no >= edge_threshold_cents:
            out.append(ArbOpportunity(market_ticker, "buy_no", edge_no, "model_vs_no_ask"))
    comp = complementary_arb_edge(yes_ask, no_ask)
    if comp > 0:
        out.append(ArbOpportunity(market_ticker, "buy_both", comp, "complementary_yes_no"))
    return out
