from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class MarketSnapshot:
    market_id: str
    event_id: Optional[str]
    timestamp: datetime
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    no_bid: Optional[float]


def normalize_orderbook(market_id: str, event_id: Optional[str], orderbook: Dict[str, Any]) -> MarketSnapshot:
    yes_bids = orderbook.get("yes", [])
    no_bids = orderbook.get("no", [])
    yes_bid = yes_bids[0][0] if yes_bids else None
    no_bid = no_bids[0][0] if no_bids else None
    yes_ask = 100 - no_bid if no_bid is not None else None
    mid = None
    spread = None
    if yes_bid is not None and yes_ask is not None:
        mid = (yes_bid + yes_ask) / 2
        spread = yes_ask - yes_bid
    ts = datetime.now(timezone.utc)
    return MarketSnapshot(
        market_id=market_id,
        event_id=event_id,
        timestamp=ts,
        bid=yes_bid,
        ask=yes_ask,
        mid=mid,
        spread=spread,
        volume=orderbook.get("volume"),
        open_interest=orderbook.get("open_interest"),
        no_bid=no_bid,
    )


def implied_probability(mid_cents: Optional[float]) -> Optional[float]:
    if mid_cents is None:
        return None
    return max(0.0, min(1.0, mid_cents / 100.0))


class ExternalSignalAdapter:
    def fetch(self, *_: Any, **__: Any) -> Dict[str, Any]:
        raise NotImplementedError("External signals require a configured adapter and API keys.")
