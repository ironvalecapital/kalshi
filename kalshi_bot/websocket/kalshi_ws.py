from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import websockets

from ..auth import build_auth_headers, load_private_key
from ..config import BotSettings, ws_url

Handler = Callable[[Dict[str, Any]], Any]


class KalshiWebsocketClient:
    """
    Persistent Kalshi WebSocket client with reconnect/backoff and dynamic subscriptions.
    """

    def __init__(self, settings: BotSettings) -> None:
        self.settings = settings
        self.url = ws_url(settings)
        self._private_key = None
        if settings.api_key_id and settings.private_key_path:
            self._private_key = load_private_key(settings.private_key_path)

        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._channels: Set[str] = set()
        self._markets: Set[str] = set()
        self._conn = None
        self._running = False
        self._book_state: Dict[str, Dict[str, Dict[int, int]]] = {}
        self._trade_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self._next_id = 1

    def _headers(self) -> Optional[Dict[str, str]]:
        if not self.settings.api_key_id or not self._private_key:
            return None
        return build_auth_headers(self.settings.api_key_id, self._private_key, "GET", "/trade-api/ws/v2")

    async def _connect(self):
        headers = self._headers()
        kwargs = dict(
            ping_interval=20,
            ping_timeout=20,
            open_timeout=self.settings.data.ws_open_timeout_sec,
        )
        if headers:
            kwargs["additional_headers"] = headers
        return await websockets.connect(self.url, **kwargs)

    def register_handler(self, channel: str, handler: Handler) -> None:
        self._handlers[channel].append(handler)

    def unregister_handler(self, channel: str, handler: Handler) -> None:
        if channel in self._handlers:
            self._handlers[channel] = [h for h in self._handlers[channel] if h is not handler]

    def get_orderbook_state(self, ticker: str) -> Dict[str, Any]:
        state = self._book_state.get(ticker, {"yes": {}, "no": {}})
        yes = sorted(state["yes"].items(), key=lambda x: x[0], reverse=True)
        no = sorted(state["no"].items(), key=lambda x: x[0], reverse=True)
        return {"yes": yes, "no": no}

    def get_trade_buffer(self, ticker: str) -> List[Dict[str, Any]]:
        return list(self._trade_buffer.get(ticker, []))

    async def subscribe(self, channels: List[str], market_tickers: Optional[List[str]] = None) -> None:
        self._channels.update(channels)
        if market_tickers:
            self._markets.update(market_tickers)
        if not self._conn:
            return
        payload: Dict[str, Any] = {"channels": sorted(self._channels)}
        if self._markets:
            payload["market_tickers"] = sorted(self._markets)
        msg = {"id": self._next_id, "cmd": "subscribe", "params": payload}
        self._next_id += 1
        await self._conn.send(json.dumps(msg))

    async def unsubscribe(self, channels: List[str], market_tickers: Optional[List[str]] = None) -> None:
        for c in channels:
            self._channels.discard(c)
        if market_tickers:
            for t in market_tickers:
                self._markets.discard(t)
        if not self._conn:
            return
        payload: Dict[str, Any] = {"channels": channels}
        if market_tickers:
            payload["market_tickers"] = market_tickers
        msg = {"id": self._next_id, "cmd": "unsubscribe", "params": payload}
        self._next_id += 1
        await self._conn.send(json.dumps(msg))

    async def _resubscribe(self) -> None:
        if not self._channels:
            return
        payload: Dict[str, Any] = {"channels": sorted(self._channels)}
        if self._markets:
            payload["market_tickers"] = sorted(self._markets)
        msg = {"id": self._next_id, "cmd": "subscribe", "params": payload}
        self._next_id += 1
        await self._conn.send(json.dumps(msg))

    def _apply_snapshot(self, ticker: str, msg: Dict[str, Any]) -> None:
        payload = msg.get("msg", msg)
        yes = {int(p): int(sz) for p, sz in (payload.get("yes") or [])}
        no = {int(p): int(sz) for p, sz in (payload.get("no") or [])}
        self._book_state[ticker] = {"yes": yes, "no": no}

    def _apply_delta(self, ticker: str, msg: Dict[str, Any]) -> None:
        payload = msg.get("msg", msg)
        side = str(payload.get("side") or "").lower()
        price = payload.get("price")
        # Kalshi deltas may include either delta or absolute size.
        delta = payload.get("delta")
        size = payload.get("size")
        if side not in {"yes", "no"} or price is None:
            return
        state = self._book_state.setdefault(ticker, {"yes": {}, "no": {}})
        book = state[side]
        p = int(price)
        if delta is not None:
            nv = int(book.get(p, 0)) + int(delta)
        elif size is not None:
            nv = int(size)
        else:
            return
        if nv <= 0:
            book.pop(p, None)
        else:
            book[p] = nv

    async def _dispatch(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type") or msg.get("channel") or "unknown"
        payload = msg.get("msg", msg)
        ticker = payload.get("market_ticker") or payload.get("ticker")
        if isinstance(ticker, str):
            ticker = ticker.strip()

        if mtype == "orderbook_snapshot" and ticker:
            self._apply_snapshot(ticker, msg)
        elif mtype == "orderbook_delta" and ticker:
            self._apply_delta(ticker, msg)
        elif mtype == "trade" and ticker:
            self._trade_buffer[ticker].append(payload)

        handlers = self._handlers.get(mtype, [])
        for handler in handlers:
            try:
                result = handler(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Keep stream alive even when strategy callback fails.
                continue

    async def run(self) -> None:
        self._running = True
        backoff = 1
        while self._running:
            try:
                async with await self._connect() as ws:
                    self._conn = ws
                    await self._resubscribe()
                    backoff = 1
                    async for raw in ws:
                        data = json.loads(raw)
                        await self._dispatch(data)
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(60, backoff * 2)
            finally:
                self._conn = None

    async def close(self) -> None:
        self._running = False
        if self._conn:
            await self._conn.close()

    async def run_with_defaults(self, market_tickers: List[str]) -> None:
        await self.subscribe(
            channels=["ticker", "ticker_v2", "orderbook_snapshot", "orderbook_delta", "trade"],
            market_tickers=market_tickers,
        )
        await self.run()

