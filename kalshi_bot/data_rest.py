from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .auth import build_auth_headers, load_private_key
from .config import BotSettings, api_base
from .rate_limit import RateLimiter

try:
    from kalshi_python import Configuration as SDKConfiguration
    from kalshi_python import KalshiClient as SDKClient
except Exception:  # pragma: no cover
    SDKConfiguration = None
    SDKClient = None


class KalshiRestError(Exception):
    def __init__(self, message: str, status_code: int | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass
class RestResponse:
    data: Dict[str, Any]
    status_code: int
    headers: Dict[str, str]


class KalshiRestClient:
    def __init__(self, settings: BotSettings, limiter: RateLimiter) -> None:
        self.settings = settings
        self.limiter = limiter
        self.base_url = api_base(settings)
        self._private_key = None
        if settings.api_key_id and settings.private_key_path:
            self._private_key = load_private_key(settings.private_key_path)
        self._client = httpx.Client(timeout=45.0)

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        if not self.settings.api_key_id or not self._private_key:
            raise KalshiRestError("Missing API key id or private key")
        sign_path = f"/trade-api/v2{path}"
        return build_auth_headers(self.settings.api_key_id, self._private_key, method, sign_path)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
        is_write: bool = False,
        retries: int = 3,
    ) -> RestResponse:
        attempt = 0
        while True:
            self.limiter.wait_for_token("write" if is_write else "read")
            url = f"{self.base_url}{path}"
            body = json.dumps(json_body) if json_body else ""
            headers = {"Content-Type": "application/json"}
            if auth:
                headers.update(self._auth_headers(method, path))
            resp = self._client.request(method, url, params=params, content=body or None, headers=headers)
            if resp.status_code in (429, 403):
                # Rate limit/backoff handling, honor Retry-After when present.
                # https://docs.kalshi.com/getting_started/rate_limits
                retry_after = None
                if "Retry-After" in resp.headers:
                    try:
                        retry_after = int(resp.headers["Retry-After"])
                    except ValueError:
                        retry_after = None
                if resp.status_code == 429 and attempt < retries:
                    backoff = retry_after if retry_after is not None else min(60, 2 ** attempt)
                    time.sleep(backoff)
                    attempt += 1
                    continue
                raise KalshiRestError(f"Rate limited or forbidden: {resp.status_code}", resp.status_code, retry_after)
            if resp.status_code >= 400:
                raise KalshiRestError(f"HTTP {resp.status_code}: {resp.text}", resp.status_code)
            return RestResponse(resp.json(), resp.status_code, dict(resp.headers))

    def list_markets(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if query:
            params["search"] = query
        # Kalshi only accepts open/closed/settled status. Ignore invalid values like "initialized".
        if status and status in {"open", "closed", "settled"}:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            params["max_close_ts"] = max_close_ts
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params).data

    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}/orderbook").data

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}").data

    def get_positions(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/positions", auth=True).data

    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", "/portfolio/orders", params=params, auth=True).data

    def create_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/portfolio/orders", json_body=payload, auth=True, is_write=True).data

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}", auth=True, is_write=True).data

    def get_trades(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets/trades", params=params).data


class KalshiSDKWrapper:
    def __init__(self, settings: BotSettings) -> None:
        if SDKClient is None or SDKConfiguration is None:
            raise KalshiRestError("kalshi-python SDK not available")
        if not settings.api_key_id or not settings.private_key_path:
            raise KalshiRestError("Missing API credentials")
        config = SDKConfiguration(host=api_base(settings))
        with open(settings.private_key_path, "r") as f:
            private_key = f.read()
        config.api_key_id = settings.api_key_id
        config.private_key_pem = private_key
        self.client = SDKClient(config)

    def get_balance(self) -> Any:
        return self.client.get_balance()


class KalshiDataClient:
    def __init__(self, settings: BotSettings, limiter: RateLimiter) -> None:
        self.rest = KalshiRestClient(settings, limiter)
        self.settings = settings
        self.sdk = None
        if settings.use_sdk and settings.api_key_id and settings.private_key_path:
            try:
                self.sdk = KalshiSDKWrapper(settings)
            except Exception:
                self.sdk = None

    def get_balance(self) -> Optional[Any]:
        if self.sdk:
            return self.sdk.get_balance()
        return None

    def list_markets(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.rest.list_markets(
            query=query,
            limit=limit,
            status=status,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
            cursor=cursor,
        )

    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        return self.rest.get_orderbook(ticker)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self.rest.get_market(ticker)

    def get_positions(self) -> Dict[str, Any]:
        return self.rest.get_positions()

    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        return self.rest.get_orders(status=status, limit=limit)

    def create_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.rest.create_order(payload)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self.rest.cancel_order(order_id)

    def get_trades(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.rest.get_trades(ticker=ticker, min_ts=min_ts, max_ts=max_ts, limit=limit, cursor=cursor)
