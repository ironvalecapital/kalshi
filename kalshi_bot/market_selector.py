from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient


@dataclass
class SportsCandidate:
    ticker: str
    title: str
    event_ticker: str
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
    ticker = str(m.get("ticker", "")).upper()
    if getattr(settings.sports, "exclude_multigame_extended", False) and "MULTIGAMEEXTENDED" in ticker:
        return False
    if getattr(settings.sports, "market_universe", "sports") == "all":
        return True
    if settings.sports.allowlist and m.get("ticker") in settings.sports.allowlist:
        return True
    text = " ".join([str(m.get("title", "")), str(m.get("subtitle", "")), str(m.get("series_ticker", ""))]).upper()
    return any(k in text for k in settings.sports.keywords)


def _orderbook_complement(ob: Dict[str, Any]) -> Dict[str, Optional[int]]:
    # Orderbook bids-only; derive asks via complement.
    # https://docs.kalshi.com/getting_started/orderbook_responses
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
    allowed_statuses = {"open", "closed", "settled", "unopened"}
    statuses = [s for s in (getattr(settings.sports, "statuses", None) or ["open"]) if s in allowed_statuses]
    if not statuses:
        statuses = ["open"]

    # Cache markets list briefly to reduce API load.
    cache_key = "sports_markets_cache"
    now_ts = time.time()
    cache = getattr(pick_sports_candidates, cache_key, None)
    if cache and (now_ts - cache["ts"] <= settings.sports.markets_cache_ttl_sec):
        markets = cache["markets"]
    else:
        for status in statuses:
            resp = data_client.list_markets(status=status, limit=min(1000, settings.sports.max_scan_markets))
            markets.extend(resp.get("markets", []))
            if len(markets) >= settings.sports.max_scan_markets:
                break
        setattr(pick_sports_candidates, cache_key, {"ts": now_ts, "markets": markets})

    # Pre-filter by volume to reduce downstream calls.
    markets = sorted(
        markets,
        key=lambda m: float(m.get("volume_24h", 0) or m.get("volume", 0) or 0),
        reverse=True,
    )
    markets = markets[: settings.sports.max_scan_markets]
    probe_markets = markets[: settings.sports.orderbook_probe_limit]
    now = datetime.now(timezone.utc)
    candidates: List[SportsCandidate] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _summary_quotes(m: Dict[str, Any]) -> Dict[str, Optional[int]]:
        yes_bid = m.get("yes_bid")
        no_bid = m.get("no_bid")
        yes_ask = m.get("yes_ask")
        no_ask = m.get("no_ask")
        if yes_ask is None and no_bid is not None:
            yes_ask = 100 - no_bid
        if no_ask is None and yes_bid is not None:
            no_ask = 100 - yes_bid
        spread_yes = (yes_ask - yes_bid) if yes_ask is not None and yes_bid is not None else None
        return {
            "best_yes_bid": yes_bid,
            "best_yes_ask": yes_ask,
            "best_no_bid": no_bid,
            "best_no_ask": no_ask,
            "spread_yes": spread_yes,
            "depth_top3": 0,
        }

    def fetch_one(m: Dict[str, Any]) -> Optional[SportsCandidate]:
        if not _is_sports(settings, m) and not settings.sports.allow_unmatched_markets:
            return None
        ob = data_client.get_orderbook(m.get("ticker"))
        prices = _orderbook_complement(ob)
        if prices["spread_yes"] is None:
            prices = _summary_quotes(m)
        if prices["spread_yes"] is None and prices["best_yes_bid"] is None and prices["best_no_bid"] is None:
            return None
        trades_resp = data_client.get_trades(ticker=m.get("ticker"), limit=200)
        trades = trades_resp.get("trades", [])
        trades_60m = _count_trades(trades, now - timedelta(minutes=60))
        trades_5m = _count_trades(trades, now - timedelta(minutes=5))
        ticker = str(m.get("ticker", "")).upper()
        if (
            "MULTIGAMEEXTENDED" in ticker
            and (prices["spread_yes"] is None or prices["spread_yes"] == 0)
            and prices["depth_top3"] <= 0
            and trades_60m <= 0
        ):
            return None
        spread_for_score = prices["spread_yes"] if prices["spread_yes"] is not None else 0
        liquidity_score = 1.0 * trades_60m + 0.5 * trades_5m + 0.1 * prices["depth_top3"] - 0.7 * spread_for_score
        return SportsCandidate(
            ticker=m.get("ticker"),
            title=m.get("title", ""),
            event_ticker=m.get("event_ticker", "") or m.get("event_id", "") or "",
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

    for m in markets:
        if not _is_sports(settings, m) and not settings.sports.allow_unmatched_markets:
            continue
        pass

    with ThreadPoolExecutor(max_workers=max(1, settings.sports.selector_workers)) as executor:
        futures = [executor.submit(fetch_one, m) for m in probe_markets]
        for fut in as_completed(futures):
            cand = fut.result()
            if cand is None:
                continue
            if "MULTIGAMEEXTENDED" in str(cand.ticker).upper():
                continue
            candidates.append(cand)
    if not candidates and getattr(settings.sports, "market_universe", "sports") == "all":
        fallback_queries = ["BTC", "ETH", "NBA", "NFL", "NCAA", "CPI", "FED", "RATE", "FINANCE"]
        for q in fallback_queries:
            try:
                resp = data_client.list_markets(query=q, status="open", limit=100)
            except Exception:
                continue
            for m in resp.get("markets", []):
                cand = fetch_one(m)
                if cand is None:
                    continue
                if "MULTIGAMEEXTENDED" in str(cand.ticker).upper():
                    continue
                candidates.append(cand)
                if len(candidates) >= top_n:
                    break
            if len(candidates) >= top_n:
                break
    candidates.sort(key=lambda c: c.liquidity_score, reverse=True)
    return candidates[:top_n]
