from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient
from .market_selector import _orderbook_complement


@dataclass
class SpreadCandidate:
    ticker: str
    title: str
    status: str
    spread_yes: Optional[int]
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    depth_top3: int
    volume_24h: float
    ts: str


def scan_spreads(
    settings: BotSettings,
    data_client: KalshiDataClient,
    top: int = 20,
    min_spread: int = 2,
    max_spread: int = 30,
    status: str = "open",
) -> List[SpreadCandidate]:
    resp = data_client.list_markets(status=status, limit=1000)
    markets = resp.get("markets", [])
    candidates: List[SpreadCandidate] = []
    now = datetime.now(timezone.utc).isoformat()
    for m in markets:
        ob = data_client.get_orderbook(m.get("ticker"))
        prices = _orderbook_complement(ob)
        spread = prices["spread_yes"]
        if spread is None:
            continue
        if spread < min_spread or spread > max_spread:
            continue
        volume = float(m.get("volume_24h", 0) or m.get("volume", 0) or 0)
        candidates.append(
            SpreadCandidate(
                ticker=m.get("ticker", ""),
                title=m.get("title", ""),
                status=m.get("status", ""),
                spread_yes=spread,
                yes_bid=prices["best_yes_bid"],
                yes_ask=prices["best_yes_ask"],
                depth_top3=prices["depth_top3"],
                volume_24h=volume,
                ts=now,
            )
        )
    candidates.sort(key=lambda c: (c.spread_yes or 0, c.volume_24h), reverse=True)
    return candidates[:top]
