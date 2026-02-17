from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, Optional

import websockets


WS_LIVE = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"


@dataclass
class L2Book:
    yes: Dict[int, int] = field(default_factory=dict)
    no: Dict[int, int] = field(default_factory=dict)
    last_ts: Optional[datetime] = None

    def _apply_side(self, side: str, price: int, size: int) -> None:
        book = self.yes if side.lower() == "yes" else self.no
        if size <= 0:
            book.pop(price, None)
        else:
            book[int(price)] = int(size)

    def apply_snapshot(self, yes_levels: Iterable, no_levels: Iterable) -> None:
        self.yes.clear()
        self.no.clear()
        for lv in yes_levels or []:
            p, s = _parse_level(lv)
            if p is not None and s is not None and s > 0:
                self.yes[p] = s
        for lv in no_levels or []:
            p, s = _parse_level(lv)
            if p is not None and s is not None and s > 0:
                self.no[p] = s
        self.last_ts = datetime.now(timezone.utc)

    def apply_delta(self, side: str, price: int, size: int) -> None:
        self._apply_side(side, price, size)
        self.last_ts = datetime.now(timezone.utc)

    def sorted_yes(self) -> list[tuple[int, int]]:
        return sorted(self.yes.items(), key=lambda x: x[0], reverse=True)

    def sorted_no(self) -> list[tuple[int, int]]:
        return sorted(self.no.items(), key=lambda x: x[0], reverse=True)


def _parse_level(level: Any) -> tuple[Optional[int], Optional[int]]:
    if isinstance(level, list) and len(level) >= 2:
        return int(level[0]), int(level[1])
    if isinstance(level, dict):
        p = level.get("price")
        s = level.get("size")
        if p is not None and s is not None:
            return int(p), int(s)
    return None, None


class KalshiWSClient:
    def __init__(self, demo: bool = False) -> None:
        self.url = WS_DEMO if demo else WS_LIVE
        self.books: Dict[str, L2Book] = defaultdict(L2Book)
        self.trades: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=5000))
        self.handlers: Dict[str, list[Callable[[Dict[str, Any]], Any]]]= defaultdict(list)

    def register_handler(self, channel: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        self.handlers[channel].append(fn)

    async def _dispatch(self, channel: str, message: Dict[str, Any]) -> None:
        for fn in self.handlers.get(channel, []):
            res = fn(message)
            if asyncio.iscoroutine(res):
                await res

    async def connect_and_stream(
        self,
        tickers: list[str],
        channels: Optional[list[str]] = None,
    ) -> None:
        channels = channels or ["orderbook_delta", "trade", "ticker_v2"]
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=10, ping_timeout=20) as ws:
                    sub = {
                        "id": int(time.time() * 1000),
                        "cmd": "subscribe",
                        "params": {"channels": channels, "market_tickers": tickers},
                    }
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        await self._process_message(msg)
            except Exception:
                await asyncio.sleep(min(30.0, backoff))
                backoff *= 2.0

    async def _process_message(self, msg: Dict[str, Any]) -> None:
        channel = (msg.get("type") or msg.get("channel") or "").lower()
        data = msg.get("data", msg)
        ticker = (
            data.get("market_ticker")
            or data.get("ticker")
            or data.get("market")
            or msg.get("market_ticker")
            or msg.get("ticker")
        )
        if channel in {"orderbook_delta", "orderbook"} and ticker:
            book = self.books[str(ticker)]
            # snapshot style
            if "yes" in data or "no" in data or "yes_bids" in data or "no_bids" in data:
                book.apply_snapshot(data.get("yes") or data.get("yes_bids") or [], data.get("no") or data.get("no_bids") or [])
            # delta style
            side = data.get("side")
            price = data.get("price")
            size = data.get("size")
            if side is not None and price is not None and size is not None:
                book.apply_delta(str(side), int(price), int(size))
        elif channel == "trade" and ticker:
            row = {
                "ts": data.get("ts") or data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "yes_price": data.get("yes_price") or data.get("price"),
                "no_price": data.get("no_price"),
                "count": data.get("count") or data.get("size") or 0,
                "taker_side": data.get("taker_side") or data.get("side"),
            }
            self.trades[str(ticker)].append(row)
        if channel:
            await self._dispatch(channel, msg)
