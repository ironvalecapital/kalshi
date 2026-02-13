from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets

from .auth import build_auth_headers, load_private_key
from .config import BotSettings, ws_url


class KalshiWSClient:
    def __init__(self, settings: BotSettings) -> None:
        self.settings = settings
        self.url = ws_url(settings)
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
            return await websockets.connect(self.url, additional_headers=headers, ping_interval=20, ping_timeout=20)
        return await websockets.connect(self.url, ping_interval=20, ping_timeout=20)

    async def subscribe(
        self,
        ws,
        channels: List[str],
        market_tickers: Optional[List[str]] = None,
        sub_id: int = 1,
    ) -> None:
        params: Dict[str, Any] = {"channels": channels}
        if market_tickers:
            params["market_tickers"] = market_tickers
        message = {"id": sub_id, "cmd": "subscribe", "params": params}
        await ws.send(json.dumps(message))

    async def stream_ticker(
        self,
        market_tickers: List[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async with await self._connect() as ws:
            await self.subscribe(ws, ["ticker"], market_tickers, sub_id=1)
            async for raw in ws:
                data = json.loads(raw)
                if data.get("type") == "ticker":
                    yield data

    async def stream_orderbook_deltas(
        self,
        market_tickers: List[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # WebSocket market data channels (ticker/orderbook).
        # https://docs.kalshi.com/getting_started/quick_start_market_data
        # https://docs.kalshi.com/api-reference/websockets
        backoff = 1
        while True:
            try:
                async with await self._connect() as ws:
                    await self.subscribe(ws, ["orderbook_snapshot", "orderbook_delta"], market_tickers, sub_id=2)
                    async for raw in ws:
                        data = json.loads(raw)
                        if data.get("type") in ("orderbook_snapshot", "orderbook_delta"):
                            yield data
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(30, backoff * 2)

    async def stream_ticker_and_book(
        self,
        market_tickers: List[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream ticker + orderbook updates with reconnect/backoff.
        """
        backoff = 1
        while True:
            try:
                async with await self._connect() as ws:
                    await self.subscribe(ws, ["ticker", "orderbook_snapshot", "orderbook_delta"], market_tickers, sub_id=3)
                    async for raw in ws:
                        data = json.loads(raw)
                        if data.get("type") in ("ticker", "orderbook_snapshot", "orderbook_delta"):
                            yield data
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(30, backoff * 2)

    async def watch(self, market_ticker: str, callback) -> None:
        async for msg in self.stream_ticker([market_ticker]):
            await callback(msg)
            await asyncio.sleep(0)
