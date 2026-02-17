from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import httpx


class KalshiRESTError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class KalshiRESTClient:
    """
    Async REST client with retries + 429 handling.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 10.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url or os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
        self.api_key = api_key or os.getenv("KALSHI_API_KEY_ID")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        # Public endpoints generally work without auth; keep simple here.
        if self.api_key:
            h["KALSHI-ACCESS-KEY"] = self.api_key
        return h

    async def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        backoff = 0.5
        for attempt in range(self.max_retries + 1):
            resp = await self._client.request(method, url, params=params, headers=self._headers())
            if resp.status_code < 400:
                return resp.json()
            if resp.status_code in (429, 503) and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else backoff
                await asyncio.sleep(min(15.0, sleep_s))
                backoff *= 2.0
                continue
            raise KalshiRESTError(resp.status_code, f"HTTP {resp.status_code}: {resp.text[:400]}")
        raise KalshiRESTError(500, "exhausted retries")

    async def list_markets(self, status: str = "open", limit: int = 200, cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/markets", params=params)

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        return await self._request("GET", f"/markets/{ticker}")

    async def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return await self._request("GET", f"/markets/{ticker}/orderbook")

    async def get_trades(self, ticker: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return await self._request("GET", "/markets/trades", params=params)

    async def get_candlesticks(self, ticker: str, period_interval: int = 1, period_unit: str = "minute") -> Dict[str, Any]:
        params = {"period_interval": period_interval, "period_unit": period_unit}
        return await self._request("GET", f"/series/{ticker}/candlesticks", params=params)
