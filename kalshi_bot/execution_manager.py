from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from .config import BotSettings
from .edge_scorer import EdgeScore, score_no, score_yes
from .execution import ExecutionEngine, OrderRequest
from .ledger import Ledger
from .market_scanner import MarketScan
from .order_lifecycle import OrderLifecycle


@dataclass
class LadderDecision:
    score: EdgeScore
    count: int


def ladder_prices(best_bid: int, best_ask: int, levels: int) -> List[int]:
    if best_bid is None or best_ask is None:
        return []
    prices = []
    for i in range(levels):
        price = best_bid + 1 + i
        if price >= best_ask:
            break
        prices.append(price)
    return prices


def maker_ladder_cycle(
    settings: BotSettings,
    exec_engine: ExecutionEngine,
    ledger: Ledger,
    scans: List[MarketScan],
    p_model: float,
    max_orders: int = 5,
    levels: int = 3,
    min_ev_cents: float = 0.0,
    slippage_cents: float = 0.0,
    lifecycle: Optional[OrderLifecycle] = None,
) -> List[LadderDecision]:
    decisions: List[LadderDecision] = []
    maker_rate = settings.execution.maker_fee_rate
    taker_rate = settings.execution.taker_fee_rate

    for scan in scans:
        if len(decisions) >= max_orders:
            break
        yes_bid = scan.best_yes_bid
        yes_ask = scan.best_yes_ask
        no_bid = scan.best_no_bid
        no_ask = scan.best_no_ask
        spread = scan.spread_yes or 0

        yes_prices = ladder_prices(yes_bid, yes_ask, levels) if yes_bid is not None and yes_ask is not None else []
        no_prices = ladder_prices(no_bid, no_ask, levels) if no_bid is not None and no_ask is not None else []

        for price in yes_prices:
            score = score_yes(
                scan.ticker,
                price_cents=price,
                p_model=p_model,
                spread_cents=spread,
                count=settings.sports.base_size,
                maker=True,
                maker_rate=maker_rate,
                taker_rate=taker_rate,
                slippage_cents=slippage_cents,
            )
            if score.ev_cents >= min_ev_cents:
                order = OrderRequest(
                    market_id=scan.ticker,
                    side="yes",
                    action="buy",
                    price_cents=price,
                    count=settings.sports.base_size,
                    client_order_id=str(uuid.uuid4()),
                )
                result = exec_engine.place_order(order)
                order_id = result.get("response", {}).get("order", {}).get("order_id")
                if lifecycle and order_id:
                    lifecycle.register(order_id, scan.ticker, "yes", price)
                decisions.append(LadderDecision(score=score, count=settings.sports.base_size))
                if len(decisions) >= max_orders:
                    break
        if len(decisions) >= max_orders:
            break

        for price in no_prices:
            score = score_no(
                scan.ticker,
                price_cents=price,
                p_model=p_model,
                spread_cents=spread,
                count=settings.sports.base_size,
                maker=True,
                maker_rate=maker_rate,
                taker_rate=taker_rate,
                slippage_cents=slippage_cents,
            )
            if score.ev_cents >= min_ev_cents:
                order = OrderRequest(
                    market_id=scan.ticker,
                    side="no",
                    action="buy",
                    price_cents=price,
                    count=settings.sports.base_size,
                    client_order_id=str(uuid.uuid4()),
                )
                result = exec_engine.place_order(order)
                order_id = result.get("response", {}).get("order", {}).get("order_id")
                if lifecycle and order_id:
                    lifecycle.register(order_id, scan.ticker, "no", price)
                decisions.append(LadderDecision(score=score, count=settings.sports.base_size))
                if len(decisions) >= max_orders:
                    break
        if len(decisions) >= max_orders:
            break

    return decisions
