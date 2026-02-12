from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient


@dataclass
class MarketCandidate:
    ticker: str
    title: str
    subtitle: str
    close_time: datetime
    best_yes_bid: Optional[int]
    best_yes_ask: Optional[int]
    best_no_bid: Optional[int]
    best_no_ask: Optional[int]
    spread_yes: Optional[int]
    trades_1h: int
    trades_24h: int
    volume: float
    liquidity_score: float
    depth_yes: int
    depth_no: int


def is_weather_market(settings: BotSettings, market: Dict[str, Any]) -> bool:
    text = " ".join([str(market.get("title", "")), str(market.get("subtitle", "")), str(market.get("series_ticker", ""))]).upper()
    return any(k in text for k in settings.weather.keywords)


def parse_close_time(market: Dict[str, Any]) -> Optional[datetime]:
    raw = market.get("close_time") or market.get("close_ts")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def orderbook_best_prices(orderbook: Dict[str, Any], max_depth_levels: int = 3) -> Dict[str, Optional[int]]:
    # Kalshi orderbook returns bids only; asks must be derived by complement.
    yes_bids = orderbook.get("yes", [])
    no_bids = orderbook.get("no", [])
    best_yes_bid = yes_bids[0][0] if yes_bids else None
    best_no_bid = no_bids[0][0] if no_bids else None
    best_yes_ask = 100 - best_no_bid if best_no_bid is not None else None
    best_no_ask = 100 - best_yes_bid if best_yes_bid is not None else None
    spread_yes = best_yes_ask - best_yes_bid if best_yes_ask is not None and best_yes_bid is not None else None
    depth_yes = sum(level[1] for level in yes_bids[:max_depth_levels]) if yes_bids else 0
    depth_no = sum(level[1] for level in no_bids[:max_depth_levels]) if no_bids else 0
    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "spread_yes": spread_yes,
        "depth_yes": depth_yes,
        "depth_no": depth_no,
    }


def count_trades_since(trades: List[Dict[str, Any]], since: datetime) -> int:
    count = 0
    for t in trades:
        ts = t.get("ts") or t.get("timestamp") or t.get("time")
        if ts is None:
            continue
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        if dt >= since:
            count += 1
    return count


def pick_weather_candidates(settings: BotSettings, data_client: KalshiDataClient, top_n: Optional[int] = None) -> List[MarketCandidate]:
    now = datetime.now(timezone.utc)
    min_close = now + timedelta(hours=settings.weather.min_time_to_close_hours)
    max_close = now + timedelta(hours=settings.weather.max_time_to_close_hours)
    markets: List[Dict[str, Any]] = []
    max_scan = settings.weather.max_scan_markets
    allowed_statuses = {"open", "closed", "settled"}
    statuses = [s for s in (getattr(settings.weather, "statuses", None) or ["open"]) if s in allowed_statuses]
    if not statuses:
        statuses = ["open"]
    for status in statuses:
        cursor = None
        while True:
            params = {
                "status": status,
                "limit": min(1000, max_scan),
                "cursor": cursor,
            }
            if settings.weather.use_close_window:
                params["min_close_ts"] = int(min_close.timestamp())
                params["max_close_ts"] = int(max_close.timestamp())
            resp = data_client.list_markets(**params)
            markets.extend(resp.get("markets", []))
            if len(markets) >= max_scan:
                break
            cursor = resp.get("cursor")
            if not cursor:
                break
        if len(markets) >= max_scan:
            break
    candidates: List[MarketCandidate] = []
    for m in markets:
        if not is_weather_market(settings, m) and not settings.weather.allow_unmatched_markets:
            continue
        close_time = parse_close_time(m)
        if not close_time:
            continue
        ob = data_client.get_orderbook(m.get("ticker"))
        prices = orderbook_best_prices(ob, settings.weather.max_depth_levels)
        if prices["spread_yes"] is None:
            continue
        trades_resp = data_client.get_trades(ticker=m.get("ticker"), limit=200)
        trades = trades_resp.get("trades", [])
        trades_1h = count_trades_since(trades, now - timedelta(hours=1))
        trades_24h = count_trades_since(trades, now - timedelta(hours=24))
        if prices["spread_yes"] > settings.weather.max_spread_cents:
            continue
        if trades_24h < settings.weather.min_trades_24h:
            continue
        volume = float(m.get("volume", 0) or 0)
        liquidity_score = 1.0 * trades_1h + 0.2 * trades_24h - 0.5 * prices["spread_yes"]
        candidates.append(
            MarketCandidate(
                ticker=m.get("ticker"),
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                close_time=close_time,
                best_yes_bid=prices["best_yes_bid"],
                best_yes_ask=prices["best_yes_ask"],
                best_no_bid=prices["best_no_bid"],
                best_no_ask=prices["best_no_ask"],
                spread_yes=prices["spread_yes"],
                trades_1h=trades_1h,
                trades_24h=trades_24h,
                volume=volume,
                liquidity_score=liquidity_score,
                depth_yes=prices["depth_yes"],
                depth_no=prices["depth_no"],
            )
        )
    candidates.sort(key=lambda c: c.liquidity_score, reverse=True)
    return candidates[: (top_n or settings.weather.top_n)]
