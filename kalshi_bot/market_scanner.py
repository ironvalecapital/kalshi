from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient
from .market_selector import _orderbook_complement


@dataclass
class MarketScan:
    ticker: str
    title: str
    status: str
    close_time: Optional[str]
    best_yes_bid: Optional[int]
    best_yes_ask: Optional[int]
    best_no_bid: Optional[int]
    best_no_ask: Optional[int]
    spread_yes: Optional[int]
    depth_top3: int
    trades_1h: int
    trades_24h: int
    volume_24h: float
    liquidity_score: float
    tradability_score: float


def _parse_close_ts(market: Dict[str, Any]) -> Optional[datetime]:
    raw = market.get("close_time") or market.get("close_ts")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def _count_trades(trades: List[Dict[str, Any]], since: datetime) -> int:
    count = 0
    for t in trades:
        ts = t.get("ts") or t.get("timestamp") or t.get("time") or t.get("created_time")
        if ts is None:
            continue
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        if dt >= since:
            count += 1
    return count


def scan_markets(
    settings: BotSettings,
    data_client: KalshiDataClient,
    top: int = 50,
    min_spread: int = 1,
    max_spread: int = 30,
    min_trades_24h: int = 1,
    min_time_to_close_min: int = 30,
    statuses: Optional[List[str]] = None,
    alpha_vol: float = 1.0,
    beta_trades_1h: float = 5.0,
    gamma_spread: float = 2.0,
) -> List[MarketScan]:
    markets: List[Dict[str, Any]] = []
    scan_statuses = statuses or ["open"]
    for status in scan_statuses:
        cursor = None
        while True:
            resp = data_client.list_markets(status=status, limit=1000, cursor=cursor)
            markets.extend(resp.get("markets", []))
            cursor = resp.get("cursor")
            if not cursor:
                break

    now = datetime.now(timezone.utc)
    scans: List[MarketScan] = []
    for m in markets:
        close_dt = _parse_close_ts(m)
        if close_dt and (close_dt - now).total_seconds() < min_time_to_close_min * 60:
            continue

        ob = data_client.get_orderbook(m.get("ticker"))
        prices = _orderbook_complement(ob)
        spread = prices["spread_yes"]
        if spread is None:
            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            if yes_bid is not None and yes_ask is not None:
                spread = yes_ask - yes_bid
                prices["best_yes_bid"] = yes_bid
                prices["best_yes_ask"] = yes_ask
                prices["best_no_bid"] = m.get("no_bid")
                prices["best_no_ask"] = m.get("no_ask")
            else:
                continue
        if spread < min_spread or spread > max_spread:
            continue

        trades_resp = data_client.get_trades(
            ticker=m.get("ticker"),
            min_ts=int((now - timedelta(hours=24)).timestamp()),
            limit=200,
        )
        trades = trades_resp.get("trades", [])
        trades_1h = _count_trades(trades, now - timedelta(hours=1))
        trades_24h = _count_trades(trades, now - timedelta(hours=24))
        if trades_24h < min_trades_24h:
            continue

        volume = float(m.get("volume_24h", 0) or m.get("volume", 0) or 0)
        liquidity = (alpha_vol * volume) + (beta_trades_1h * trades_1h) - (gamma_spread * spread)
        tradability = liquidity + (0.2 * trades_24h) - (0.5 * spread)
        scans.append(
            MarketScan(
                ticker=m.get("ticker", ""),
                title=m.get("title", ""),
                status=m.get("status", ""),
                close_time=close_dt.isoformat() if close_dt else None,
                best_yes_bid=prices.get("best_yes_bid"),
                best_yes_ask=prices.get("best_yes_ask"),
                best_no_bid=prices.get("best_no_bid"),
                best_no_ask=prices.get("best_no_ask"),
                spread_yes=spread,
                depth_top3=prices.get("depth_top3", 0),
                trades_1h=trades_1h,
                trades_24h=trades_24h,
                volume_24h=volume,
                liquidity_score=liquidity,
                tradability_score=tradability,
            )
        )

    scans.sort(key=lambda s: s.tradability_score, reverse=True)
    return scans[:top]
