from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class NWSPoint:
    forecast_hourly: str
    forecast_grid: str


@dataclass
class HourlyPeriod:
    start_time: datetime
    temperature_f: float


class NWSClient:
    def __init__(self, user_agent: str, timeout: float = 15.0) -> None:
        # NWS API requires a User-Agent; /points discovers forecast endpoints.
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent})
        self.cache: Dict[Tuple[float, float, int], Any] = {}

    def points(self, lat: float, lon: float) -> NWSPoint:
        url = f"https://api.weather.gov/points/{lat},{lon}"
        data = self._get_json(url)
        props = data["properties"]
        return NWSPoint(forecast_hourly=props["forecastHourly"], forecast_grid=props["forecastGridData"])

    def hourly_forecast(self, url: str) -> List[HourlyPeriod]:
        data = self._get_json(url)
        periods = data["properties"]["periods"]
        out: List[HourlyPeriod] = []
        for p in periods:
            ts = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")).astimezone(timezone.utc)
            out.append(HourlyPeriod(start_time=ts, temperature_f=float(p["temperature"])))
        return out

    def daily_high_probabilities(
        self,
        hourly: List[HourlyPeriod],
        date_local: datetime,
        threshold_f: Optional[float] = None,
        bucket: Optional[Tuple[float, float]] = None,
    ) -> float:
        mu, sigma = compute_daily_high_distribution(hourly, date_local)
        if threshold_f is not None:
            return prob_high_ge(mu, sigma, threshold_f)
        if bucket:
            return prob_high_in_bucket(mu, sigma, bucket[0], bucket[1])
        return 0.0

    def _get_json(self, url: str, retries: int = 3) -> Dict[str, Any]:
        attempt = 0
        while True:
            resp = self.client.get(url)
            if resp.status_code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(min(10, 2 ** attempt))
                attempt += 1
                continue
            resp.raise_for_status()
            return resp.json()


def compute_daily_high_distribution(hourly: List[HourlyPeriod], date_local: datetime) -> Tuple[float, float]:
    temps = [h.temperature_f for h in hourly]
    if not temps:
        return 0.0, 5.0
    mu = max(temps)
    hours_ahead = max(1.0, (date_local.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 3600.0)
    sigma = 2.5 + min(4.0, hours_ahead / 12.0)
    return mu, sigma


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def prob_high_ge(mu: float, sigma: float, threshold_f: float) -> float:
    return 1.0 - normal_cdf(threshold_f, mu, sigma)


def prob_high_in_bucket(mu: float, sigma: float, low_f: float, high_f: float) -> float:
    return max(0.0, normal_cdf(high_f, mu, sigma) - normal_cdf(low_f, mu, sigma))
