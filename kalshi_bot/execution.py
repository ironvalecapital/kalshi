from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import ExecutionConfig
from .data_rest import KalshiDataClient
from .ledger import Ledger
from .risk import RiskManager
from .ev import estimate_fee_cents


@dataclass
class OrderRequest:
    market_id: str
    side: str
    action: str
    price_cents: int
    count: int
    client_order_id: str


class ExecutionEngine:
    def __init__(
        self,
        data_client: KalshiDataClient,
        ledger: Ledger,
        risk: RiskManager,
        config: ExecutionConfig,
    ) -> None:
        self.data_client = data_client
        self.ledger = ledger
        self.risk = risk
        self.config = config
        self.last_order_ts: Dict[str, float] = {}
        self.cancel_timestamps: list[float] = []

    def _kill_switch(self) -> bool:
        return os.getenv("KALSHI_BOT_KILL", "0") in ("1", "true", "TRUE")

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        if self._kill_switch():
            raise RuntimeError("Kill switch enabled")
        notional = order.price_cents * order.count / 100.0
        ok, reason = self.risk.check_order(order.market_id, order.count, notional)
        if not ok:
            return {"status": "rejected", "reason": reason}
        last_ts = self.last_order_ts.get(order.market_id)
        if last_ts and time.time() - last_ts < self.config.min_order_interval_seconds:
            return {"status": "rejected", "reason": "min_order_interval"}
        payload = {
            "ticker": order.market_id,
            "action": order.action,
            "side": order.side,
            "count": order.count,
            "type": "limit",
            "client_order_id": order.client_order_id,
            "post_only": self.config.prefer_maker,
        }
        if order.side == "yes":
            payload["yes_price"] = order.price_cents
        else:
            payload["no_price"] = order.price_cents
        response = self.data_client.create_order(payload)
        self.last_order_ts[order.market_id] = time.time()
        self.ledger.record_order(order.market_id, order.side, order.action, order.price_cents, order.count, response)
        return {"status": "submitted", "response": response}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        now = time.time()
        self.cancel_timestamps = [t for t in self.cancel_timestamps if now - t < 60]
        if len(self.cancel_timestamps) >= self.config.max_cancels_per_minute:
            return {"status": "rejected", "reason": "max_cancel_rate_exceeded"}
        response = self.data_client.cancel_order(order_id)
        self.ledger.record_audit("order_cancel", f"cancelled {order_id}", {"response": response})
        self.cancel_timestamps.append(now)
        return response
