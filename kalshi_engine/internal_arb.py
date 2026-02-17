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


@dataclass
class StructuralConstraintOpportunity:
    constraint: str
    lhs: float
    rhs: float
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


def yes_no_bid_sum_opportunity(
    market_ticker: str,
    yes_bid_cents: Optional[int],
    no_bid_cents: Optional[int],
    min_edge_cents: float = 1.0,
) -> Optional[ArbOpportunity]:
    """
    Internal structural check on same market bid books:
    - yes_bid + no_bid < 100: synthetic underpricing (buy both sides if executable)
    - yes_bid + no_bid > 100: synthetic overpricing (sell both sides if executable)
    """
    if yes_bid_cents is None or no_bid_cents is None:
        return None
    s = float(yes_bid_cents + no_bid_cents)
    if s < (100.0 - min_edge_cents):
        return ArbOpportunity(
            market_ticker=market_ticker,
            side="buy_both",
            edge_cents=100.0 - s,
            reason="yes_no_bid_sum_under_100",
        )
    if s > (100.0 + min_edge_cents):
        return ArbOpportunity(
            market_ticker=market_ticker,
            side="sell_both",
            edge_cents=s - 100.0,
            reason="yes_no_bid_sum_over_100",
        )
    return None


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


def nested_probability_opportunity(
    parent_label: str,
    parent_prob: float,
    child_label: str,
    child_prob: float,
    min_gap: float = 0.01,
) -> Optional[StructuralConstraintOpportunity]:
    """
    For nested outcomes: P(child) must be <= P(parent).
    Example: P(team wins by >3) <= P(team wins)
    """
    p = max(0.0, min(1.0, float(parent_prob)))
    c = max(0.0, min(1.0, float(child_prob)))
    gap = c - p
    if gap <= min_gap:
        return None
    return StructuralConstraintOpportunity(
        constraint=f"P({child_label}) <= P({parent_label})",
        lhs=c,
        rhs=p,
        gap=gap,
        action="sell_child_buy_parent",
        reason="nested_probability_violation",
    )


def time_derivative_opportunity(
    intraday_label: str,
    intraday_prob: float,
    close_label: str,
    close_prob: float,
    min_gap: float = 0.01,
) -> Optional[StructuralConstraintOpportunity]:
    """
    Temporal consistency check requested by strategy:
    P(intraday > X) <= P(close > X)
    """
    i = max(0.0, min(1.0, float(intraday_prob)))
    c = max(0.0, min(1.0, float(close_prob)))
    gap = i - c
    if gap <= min_gap:
        return None
    return StructuralConstraintOpportunity(
        constraint=f"P({intraday_label}) <= P({close_label})",
        lhs=i,
        rhs=c,
        gap=gap,
        action="sell_intraday_buy_close",
        reason="time_derivative_violation",
    )


def spread_compression_signal(
    spread_cents: Optional[float],
    spread_baseline_cents: float,
    no_fundamental_update: bool,
    depth_total_top5: int,
    thin_depth_threshold: int = 150,
    widen_mult: float = 1.8,
) -> Optional[ArbOpportunity]:
    """
    Signal for liquidity-provision reversion:
    spread is unusually wide, book is thin, and no fresh fundamental catalyst.
    """
    if spread_cents is None or spread_baseline_cents <= 0 or not no_fundamental_update:
        return None
    if depth_total_top5 > thin_depth_threshold:
        return None
    if float(spread_cents) < float(spread_baseline_cents) * float(widen_mult):
        return None
    return ArbOpportunity(
        market_ticker="MULTI",
        side="post_both_sides",
        edge_cents=float(spread_cents) - float(spread_baseline_cents),
        reason="spread_compression_reversion",
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
