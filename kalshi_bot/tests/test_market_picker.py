from datetime import datetime, timedelta, timezone

from kalshi_bot.config import BotSettings
from kalshi_bot.market_picker import pick_weather_candidates


class DummyClient:
    def list_markets(self, **kwargs):
        now = datetime.now(timezone.utc)
        return {
            "markets": [
                {
                    "ticker": "WX1",
                    "title": "NYC High Temperature ≥ 50",
                    "subtitle": "Weather",
                    "series_ticker": "WEATHER",
                    "close_time": (now + timedelta(hours=12)).isoformat(),
                    "volume": 10,
                }
            ],
            "cursor": None,
        }

    def get_orderbook(self, ticker):
        return {"yes": [[45, 10]], "no": [[52, 10]]}

    def get_trades(self, **kwargs):
        now = datetime.now(timezone.utc)
        return {
            "trades": [
                {"ts": (now - timedelta(hours=1)).isoformat()},
                {"ts": (now - timedelta(hours=2)).isoformat()},
                {"ts": (now - timedelta(hours=3)).isoformat()},
                {"ts": (now - timedelta(hours=4)).isoformat()},
                {"ts": (now - timedelta(hours=5)).isoformat()},
                {"ts": (now - timedelta(hours=6)).isoformat()},
            ]
        }


def test_market_picker_filters_and_ranks():
    settings = BotSettings()
    settings.weather.min_trades_24h = 5
    settings.weather.max_spread_cents = 10
    candidates = pick_weather_candidates(settings, DummyClient(), top_n=10)
    assert len(candidates) == 1
    assert candidates[0].ticker == "WX1"
