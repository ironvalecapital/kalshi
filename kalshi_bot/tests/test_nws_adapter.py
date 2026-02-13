from datetime import datetime, timezone

import pytest

httpx = pytest.importorskip("httpx")

from kalshi_bot.adapters.nws import NWSClient


class DummyResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)


def test_nws_points_and_hourly(monkeypatch):
    def fake_get(self, url):
        if "/points/" in url:
            return DummyResponse(
                {
                    "properties": {
                        "forecastHourly": "https://api.weather.gov/gridpoints/OKX/1,2/forecast/hourly",
                        "forecastGridData": "https://api.weather.gov/gridpoints/OKX/1,2",
                    }
                }
            )
        return DummyResponse(
            {
                "properties": {
                    "periods": [
                        {"startTime": "2026-02-10T10:00:00Z", "temperature": 40},
                        {"startTime": "2026-02-10T11:00:00Z", "temperature": 42},
                    ]
                }
            }
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get, raising=False)
    client = NWSClient(user_agent="test-agent")
    point = client.points(40.0, -70.0)
    assert point.forecast_hourly.startswith("https://api.weather.gov/")
    hourly = client.hourly_forecast(point.forecast_hourly)
    p = client.daily_high_probabilities(hourly, datetime.now(timezone.utc), threshold_f=41)
    assert 0.0 <= p <= 1.0
