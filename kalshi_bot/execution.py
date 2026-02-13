from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import ExecutionConfig
from .data_rest import KalshiDataClient, KalshiRestError
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

    def _extract_numeric(self, obj: Any) -> list[float]:
        vals: list[float] = []
        if isinstance(obj, (int, float)):
            vals.append(float(obj))
            return vals
        if isinstance(obj, str):
            try:
                vals.append(float(obj))
            except ValueError:
                pass
            return vals
        if isinstance(obj, dict):
            for v in obj.values():
                vals.extend(self._extract_numeric(v))
            return vals
        if isinstance(obj, list):
            for v in obj:
                vals.extend(self._extract_numeric(v))
            return vals
        return vals

    def _available_balance_usd(self) -> Optional[float]:
        bal = self.data_client.get_balance()
        if bal is None:
            return None
        # Try common balance keys first; then fallback to max positive numeric.
        candidates: list[float] = []
        if isinstance(bal, dict):
            for key in ("available_balance", "available", "cash", "balance", "buying_power"):
                if key in bal:
                    candidates.extend(self._extract_numeric(bal.get(key)))
        candidates.extend(self._extract_numeric(bal))
        positives = [x for x in candidates if x >= 0]
        if not positives:
            return None
        return max(positives)

    def _kill_switch(self) -> bool:
        return os.getenv("KALSHI_BOT_KILL", "0") in ("1", "true", "TRUE")

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        if self._kill_switch():
            raise RuntimeError("Kill switch enabled")
        notional = order.price_cents * order.count / 100.0
        signed_notional = -notional if order.action == "sell" else notional
        count = order.count
        if order.action == "buy":
            available = self._available_balance_usd()
            if available is not None:
                affordable = int(available * 100 // max(1, order.price_cents))
                if affordable <= 0:
                    return {"status": "rejected", "reason": "insufficient_balance_precheck"}
                count = min(count, affordable)
        ok, reason = self.risk.check_order(order.market_id, count, signed_notional)
        if not ok:
            return {"status": "rejected", "reason": reason}
        last_ts = self.last_order_ts.get(order.market_id)
        if last_ts and time.time() - last_ts < self.config.min_order_interval_seconds:
            return {"status": "rejected", "reason": "min_order_interval"}
        payload = {
            "ticker": order.market_id,
            "action": order.action,
            "side": order.side,
            "count": count,
            "type": "limit",
            "client_order_id": order.client_order_id,
            "post_only": self.config.prefer_maker,
        }
        if order.side == "yes":
            payload["yes_price"] = order.price_cents
        else:
            payload["no_price"] = order.price_cents
        try:
            response = self.data_client.create_order(payload)
        except KalshiRestError as exc:
            msg = str(exc).lower()
            if exc.status_code == 400 and "insufficient balance" in msg:
                return {"status": "rejected", "reason": "insufficient_balance"}
            return {"status": "rejected", "reason": f"api_error_{exc.status_code or 'unknown'}"}
        self.last_order_ts[order.market_id] = time.time()
        self.ledger.record_order(order.market_id, order.side, order.action, order.price_cents, count, response)
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
