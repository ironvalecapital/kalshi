from __future__ import annotations

from typing import Any, Dict, List

import httpx


class PolygonClient:
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=30.0))

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        date_from: str,
        date_to: str,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
    ) -> List[Dict[str, Any]]:
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{date_from}/{date_to}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
            "apiKey": self.api_key,
        }
        resp = self._client.get(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) or []
