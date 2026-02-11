from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient


@dataclass
class SportsCandidate:
    ticker: str
    title: str
    close_time: Optional[datetime]
    best_yes_bid: Optional[int]
    best_yes_ask: Optional[int]
    best_no_bid: Optional[int]
    best_no_ask: Optional[int]
    spread_yes: Optional[int]
    trades_60m: int
    trades_5m: int
    depth_top3: int
    liquidity_score: float


def _close_time(market: Dict[str, Any]) -> Optional[datetime]:
    raw = market.get("close_time") or market.get("close_ts")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def _is_sports(settings: BotSettings, m: Dict[str, Any]) -> bool:
    if settings.sports.allowlist and m.get("ticker") in settings.sports.allowlist:
        return True
    text = " ".join([str(m.get("title", "")), str(m.get("subtitle", "")), str(m.get("series_ticker", ""))]).upper()
    return any(k in text for k in settings.sports.keywords)


def _orderbook_complement(ob: Dict[str, Any]) -> Dict[str, Optional[int]]:
    # Orderbook bids-only; derive asks via complement.
    # https://docs.kalshi.com/api-reference/markets/get-market-orderbook
    yes_bids = ob.get("yes", [])
    no_bids = ob.get("no", [])
    best_yes_bid = yes_bids[0][0] if yes_bids else None
    best_no_bid = no_bids[0][0] if no_bids else None
    best_yes_ask = 100 - best_no_bid if best_no_bid is not None else None
    best_no_ask = 100 - best_yes_bid if best_yes_bid is not None else None
    spread_yes = best_yes_ask - best_yes_bid if best_yes_ask is not None and best_yes_bid is not None else None
    depth_yes = sum(level[1] for level in yes_bids[:3]) if yes_bids else 0
    depth_no = sum(level[1] for level in no_bids[:3]) if no_bids else 0
    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "spread_yes": spread_yes,
        "depth_top3": depth_yes + depth_no,
    }


def _count_trades(trades: List[Dict[str, Any]], since: datetime) -> int:
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


def pick_sports_candidates(settings: BotSettings, data_client: KalshiDataClient, top_n: int) -> List[SportsCandidate]:
    markets: List[Dict[str, Any]] = []
    allowed_statuses = {"open", "closed", "settled"}
    statuses = [s for s in (settings.sports.statuses or ["open"]) if s in allowed_statuses]
    if not statuses:
        statuses = ["open"]
    for status in statuses:
        resp = data_client.list_markets(status=status, limit=1000)
        markets.extend(resp.get("markets", []))
    now = datetime.now(timezone.utc)
    candidates: List[SportsCandidate] = []
    for m in markets:
        if not _is_sports(settings, m) and not settings.sports.allow_unmatched_markets:
            continue
        ob = data_client.get_orderbook(m.get("ticker"))
        prices = _orderbook_complement(ob)
        if prices["spread_yes"] is None:
            continue
        # Trades tape from official Get Trades endpoint.
        # https://docs.kalshi.com/api-reference/markets/get-trades
        trades_resp = data_client.get_trades(ticker=m.get("ticker"), limit=200)
        trades = trades_resp.get("trades", [])
        trades_60m = _count_trades(trades, now - timedelta(minutes=60))
        trades_5m = _count_trades(trades, now - timedelta(minutes=5))
        if prices["spread_yes"] > settings.sports.max_spread_cents:
            continue
        if trades_60m < settings.sports.min_trades_60m:
            continue
        if trades_5m < settings.sports.min_trades_5m:
            continue
        if prices["depth_top3"] < settings.sports.min_top_depth:
            continue
        liquidity_score = 1.0 * trades_60m + 0.5 * trades_5m + 0.1 * prices["depth_top3"] - 0.7 * prices["spread_yes"]
        candidates.append(
            SportsCandidate(
                ticker=m.get("ticker"),
                title=m.get("title", ""),
                close_time=_close_time(m),
                best_yes_bid=prices["best_yes_bid"],
                best_yes_ask=prices["best_yes_ask"],
                best_no_bid=prices["best_no_bid"],
                best_no_ask=prices["best_no_ask"],
                spread_yes=prices["spread_yes"],
                trades_60m=trades_60m,
                trades_5m=trades_5m,
                depth_top3=prices["depth_top3"],
                liquidity_score=liquidity_score,
            )
        )
    candidates.sort(key=lambda c: c.liquidity_score, reverse=True)
    return candidates[:top_n]
