from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import websockets

from .auth import build_auth_headers, load_private_key
from .config import BotSettings, ws_url


@dataclass
class OrderbookState:
    yes_bids: Dict[int, int]
    no_bids: Dict[int, int]

    def best_yes_bid(self) -> Optional[int]:
        return max(self.yes_bids) if self.yes_bids else None

    def best_no_bid(self) -> Optional[int]:
        return max(self.no_bids) if self.no_bids else None

    def best_yes_ask(self) -> Optional[int]:
        # Orderbook is bids-only; asks are derived via complement.
        # https://docs.kalshi.com/api-reference/markets/get-market-orderbook
        best_no = self.best_no_bid()
        return 100 - best_no if best_no is not None else None

    def best_no_ask(self) -> Optional[int]:
        best_yes = self.best_yes_bid()
        return 100 - best_yes if best_yes is not None else None

    def spread_yes(self) -> Optional[int]:
        best_yes = self.best_yes_bid()
        best_yes_ask = self.best_yes_ask()
        if best_yes is None or best_yes_ask is None:
            return None
        return best_yes_ask - best_yes

    def depth_topk(self, k: int = 3) -> int:
        yes_levels = sorted(self.yes_bids.items(), key=lambda x: x[0], reverse=True)[:k]
        no_levels = sorted(self.no_bids.items(), key=lambda x: x[0], reverse=True)[:k]
        return sum(size for _, size in yes_levels) + sum(size for _, size in no_levels)

    def depth_yes_topk(self, k: int = 3) -> int:
        yes_levels = sorted(self.yes_bids.items(), key=lambda x: x[0], reverse=True)[:k]
        return sum(size for _, size in yes_levels)

    def depth_no_topk(self, k: int = 3) -> int:
        no_levels = sorted(self.no_bids.items(), key=lambda x: x[0], reverse=True)[:k]
        return sum(size for _, size in no_levels)


class LiveOrderbook:
    def __init__(self, settings: BotSettings, ticker: str) -> None:
        self.settings = settings
        self.ticker = ticker
        self.state = OrderbookState(yes_bids={}, no_bids={})
        self._private_key = None
        if settings.api_key_id and settings.private_key_path:
            self._private_key = load_private_key(settings.private_key_path)

    def _headers(self) -> Optional[Dict[str, str]]:
        if not self.settings.api_key_id or not self._private_key:
            return None
        return build_auth_headers(self.settings.api_key_id, self._private_key, "GET", "/trade-api/ws/v2")

    async def _connect(self):
        headers = self._headers()
        if headers:
            return await websockets.connect(ws_url(self.settings), additional_headers=headers, ping_interval=20, ping_timeout=20)
        return await websockets.connect(ws_url(self.settings), ping_interval=20, ping_timeout=20)

    async def subscribe(self, ws) -> None:
        # WebSocket market data channels: orderbook_snapshot then orderbook_delta.
        # https://docs.kalshi.com/getting_started/quick_start_market_data
        # https://docs.kalshi.com/api-reference/websockets/orderbook-updates
        message = {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_snapshot", "orderbook_delta"], "market_tickers": [self.ticker]},
        }
        await ws.send(json.dumps(message))

    def apply_snapshot(self, msg: Dict[str, Any]) -> None:
        ob = msg.get("msg", msg)
        self.state.yes_bids = {int(p): int(sz) for p, sz in ob.get("yes", [])}
        self.state.no_bids = {int(p): int(sz) for p, sz in ob.get("no", [])}

    def apply_delta(self, msg: Dict[str, Any]) -> None:
        delta = msg.get("msg", msg)
        side = delta.get("side")
        price = int(delta.get("price"))
        size = int(delta.get("size"))
        book = self.state.yes_bids if side == "yes" else self.state.no_bids
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

    async def run(self, on_update) -> None:
        async with await self._connect() as ws:
            await self.subscribe(ws)
            async for raw in ws:
                data = json.loads(raw)
                if data.get("type") == "orderbook_snapshot":
                    self.apply_snapshot(data)
                    await on_update(self.state)
                if data.get("type") == "orderbook_delta":
                    self.apply_delta(data)
                    await on_update(self.state)
