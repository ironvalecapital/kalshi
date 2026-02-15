from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class CryptoPulse:
    symbol: str
    price_usd: float
    change_1h_pct: float
    change_24h_pct: float
    volume_24h: float


class CoinGeckoClient:
    """
    Free public market data adapter.
    Docs: https://www.coingecko.com/en/api/documentation
    """

    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3", timeout: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_pulse(self, coin_id: str = "bitcoin") -> Optional[CryptoPulse]:
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": coin_id,
            "price_change_percentage": "1h,24h",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        if not payload:
            return None
        row = payload[0]
        return CryptoPulse(
            symbol=str(row.get("symbol", coin_id)).upper(),
            price_usd=float(row.get("current_price") or 0.0),
            change_1h_pct=float(row.get("price_change_percentage_1h_in_currency") or 0.0),
            change_24h_pct=float(row.get("price_change_percentage_24h_in_currency") or 0.0),
            volume_24h=float(row.get("total_volume") or 0.0),
        )
