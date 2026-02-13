import time

import pytest

httpx = pytest.importorskip("httpx")

from kalshi_bot.config import BotSettings
from kalshi_bot.data_rest import KalshiRestClient
from kalshi_bot.rate_limit import RateLimiter


class SeqClient:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def request(self, method, url, params=None, content=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def test_backoff_retry_after(monkeypatch):
    settings = BotSettings()
    limiter = RateLimiter(1000, 1000, 1000)
    rest = KalshiRestClient(settings, limiter)

    responses = [
        httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "rate limit"}),
        httpx.Response(200, json={"ok": True}),
    ]
    rest._client = SeqClient(responses)

    sleeps = []

    def fake_sleep(x):
        sleeps.append(x)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    resp = rest._request("GET", "/markets", params={"limit": 1})
    assert resp.data.get("ok") is True
    assert rest._client.calls == 2
    assert sleeps and sleeps[0] == 1
