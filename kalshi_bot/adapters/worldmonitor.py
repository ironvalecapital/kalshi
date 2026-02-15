from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class WorldNewsItem:
    title: str
    url: Optional[str]
    published_at: Optional[str]
    source: Optional[str]
    sentiment: Optional[float]
    raw: Dict[str, Any]


class WorldMonitorClient:
    """
    Generic global-news adapter.
    Supports providers using either header auth or query-string API key.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        header_name: str = "x-api-key",
        timeout: float = 12.0,
        default_path: str = "/search-news",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.header_name = header_name
        self.timeout = timeout
        self.default_path = default_path

    def _extract_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("news", "articles", "items", "results", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                inner = val.get("items")
                if isinstance(inner, list):
                    return inner
        return []

    def _normalize(self, row: Dict[str, Any]) -> WorldNewsItem:
        title = str(row.get("title") or row.get("headline") or row.get("name") or "").strip()
        url = row.get("url") or row.get("link")
        published = row.get("publish_date") or row.get("publishedAt") or row.get("published_at") or row.get("date")
        source = row.get("source") or row.get("source_name")
        if isinstance(source, dict):
            source = source.get("name")
        sentiment = row.get("sentiment")
        if sentiment is not None:
            try:
                sentiment = float(sentiment)
            except Exception:
                sentiment = None
        return WorldNewsItem(
            title=title,
            url=str(url) if url else None,
            published_at=str(published) if published else None,
            source=str(source) if source else None,
            sentiment=sentiment,
            raw=row,
        )

    def search_news(
        self,
        query: str,
        limit: int = 20,
        path: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[WorldNewsItem]:
        endpoint = path or self.default_path
        url = f"{self.base_url}{endpoint if endpoint.startswith('/') else '/' + endpoint}"
        params: Dict[str, Any] = {"q": query, "query": query, "text": query, "limit": limit, "number": limit}
        if language:
            params["language"] = language
            params["lang"] = language
        if country:
            params["country"] = country
            params["source-country"] = country

        headers = {self.header_name: self.api_key}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params, headers=headers)
            # Fallback for APIs expecting query-string key (e.g., ?api-key=...)
            if resp.status_code in (401, 403):
                alt_params = dict(params)
                alt_params["api-key"] = self.api_key
                resp = client.get(url, params=alt_params)
            resp.raise_for_status()
            payload = resp.json()

        rows = self._extract_items(payload)
        out = [self._normalize(r) for r in rows if isinstance(r, dict)]
        return out[:limit]

    def market_news_score(self, text: str, limit: int = 20) -> Dict[str, Any]:
        query = " ".join([x for x in (text or "").replace("/", " ").split() if len(x) > 2][:8])
        items = self.search_news(query=query, limit=limit)
        if not items:
            return {"query": query, "count": 0, "sentiment_avg": None, "latest_ts": None}
        sentiments = [i.sentiment for i in items if i.sentiment is not None]
        sentiment_avg = sum(sentiments) / len(sentiments) if sentiments else None
        latest_ts: Optional[str] = None
        for i in items:
            ts = i.published_at
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                if latest_ts is None or dt.isoformat() > latest_ts:
                    latest_ts = dt.isoformat()
            except Exception:
                continue
        return {
            "query": query,
            "count": len(items),
            "sentiment_avg": sentiment_avg,
            "latest_ts": latest_ts,
            "headlines": [{"title": i.title, "url": i.url, "source": i.source, "published_at": i.published_at} for i in items[:5]],
        }

