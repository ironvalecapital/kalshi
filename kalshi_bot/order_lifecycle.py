from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .execution import ExecutionEngine
from .ledger import Ledger


@dataclass
class ManagedOrder:
    order_id: str
    ticker: str
    side: str
    price_cents: int
    created_ts: float


class OrderLifecycle:
    def __init__(self, exec_engine: ExecutionEngine, ledger: Ledger) -> None:
        self.exec_engine = exec_engine
        self.ledger = ledger
        self.open_orders: Dict[str, ManagedOrder] = {}

    def register(self, order_id: str, ticker: str, side: str, price_cents: int) -> None:
        self.open_orders[order_id] = ManagedOrder(
            order_id=order_id,
            ticker=ticker,
            side=side,
            price_cents=price_cents,
            created_ts=time.time(),
        )

    def list_open(self) -> List[ManagedOrder]:
        return list(self.open_orders.values())

    def cancel_stale(self, max_age_sec: int, reason: str = "stale") -> int:
        now = time.time()
        cancelled = 0
        for order_id, order in list(self.open_orders.items()):
            if now - order.created_ts >= max_age_sec:
                self.exec_engine.cancel_order(order_id)
                self.ledger.record_audit("order_cancel", reason, {"order_id": order_id, "ticker": order.ticker})
                self.open_orders.pop(order_id, None)
                cancelled += 1
        return cancelled

    def drop_filled(self, order_ids: Optional[List[str]] = None) -> None:
        if not order_ids:
            return
        for order_id in order_ids:
            self.open_orders.pop(order_id, None)
