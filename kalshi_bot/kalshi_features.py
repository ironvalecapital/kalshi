from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from .config import BotSettings
from .data_rest import KalshiDataClient
from .data_ws import KalshiWSClient


def _to_dt(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    txt = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(txt).astimezone(timezone.utc)
    except Exception:
        return None


def _best_price(levels: Any) -> Optional[int]:
    if not isinstance(levels, list) or not levels:
        return None
    row = levels[0]
    if isinstance(row, list) and row:
        return int(row[0])
    return None


def _depth_within(levels: Any, best_bid: Optional[int], cents: int) -> int:
    if best_bid is None or not isinstance(levels, list):
        return 0
    lo = best_bid - cents
    total = 0
    for lv in levels:
        if not isinstance(lv, list) or len(lv) < 2:
            continue
        p, sz = int(lv[0]), int(lv[1] or 0)
        if p >= lo:
            total += max(0, sz)
    return total


def _vwap_depth(levels: Any, best_bid: Optional[int], cents: int) -> Optional[float]:
    if best_bid is None or not isinstance(levels, list):
        return None
    lo = best_bid - cents
    notional = 0.0
    qty = 0
    for lv in levels:
        if not isinstance(lv, list) or len(lv) < 2:
            continue
        p, sz = int(lv[0]), int(lv[1] or 0)
        if p >= lo and sz > 0:
            notional += p * sz
            qty += sz
    if qty <= 0:
        return None
    return notional / qty


def implied_prob(price_cents: Optional[float]) -> Optional[float]:
    if price_cents is None:
        return None
    return max(0.0, min(1.0, float(price_cents) / 100.0))


@dataclass
class MarketMetadataFeatures:
    ticker: str
    event_ticker: str
    status: str
    yes_price: Optional[float]
    no_price: Optional[float]
    mid_price: Optional[float]
    close_time: Optional[str]
    time_to_close_sec: Optional[float]
    volume_24h: float
    last_updated: Optional[str]


@dataclass
class OrderbookFeatures:
    best_yes_bid: Optional[int]
    best_yes_ask: Optional[int]
    best_no_bid: Optional[int]
    best_no_ask: Optional[int]
    spread_yes: Optional[float]
    spread_no: Optional[float]
    avg_spread: Optional[float]
    depth_yes_within_cents: int
    depth_no_within_cents: int
    vwap_yes_depth: Optional[float]
    vwap_no_depth: Optional[float]


@dataclass
class TradeFeatures:
    trade_count_1h: int
    trade_count_24h: int
    trade_velocity: float
    avg_trade_price: Optional[float]
    trade_price_dispersion: Optional[float]


@dataclass
class ProbabilityMomentumFeatures:
    p_yes: Optional[float]
    p_no: Optional[float]
    dp_yes: Optional[float]
    dp_no: Optional[float]
    ema_yes: Optional[float]
    ema_no: Optional[float]


@dataclass
class LiquidityEdgeFeatures:
    liquidity_score: Optional[float]
    p_gap_yes: Optional[float]
    p_gap_no: Optional[float]
    eu_yes: Optional[float]
    eu_no: Optional[float]


@dataclass
class ConsistencyFeatures:
    sum_of_p_yes: Optional[float]
    inconsistency: Optional[float]


@dataclass
class KalshiFeatureVector:
    market: MarketMetadataFeatures
    orderbook: OrderbookFeatures
    trades: TradeFeatures
    probability: ProbabilityMomentumFeatures
    liquidity: LiquidityEdgeFeatures
    consistency: ConsistencyFeatures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market": asdict(self.market),
            "orderbook": asdict(self.orderbook),
            "trades": asdict(self.trades),
            "probability": asdict(self.probability),
            "liquidity": asdict(self.liquidity),
            "consistency": asdict(self.consistency),
        }


class KalshiFeatureEngineer:
    def __init__(self, settings: BotSettings, data_client: KalshiDataClient) -> None:
        self.settings = settings
        self.data_client = data_client
        self.ws = KalshiWSClient(settings)
        self._prev_p_yes: Dict[str, float] = {}
        self._prev_p_no: Dict[str, float] = {}
        self._ema_yes: Dict[str, float] = {}
        self._ema_no: Dict[str, float] = {}
        self._book_cache: Dict[str, Dict[str, Any]] = {}
        self._trade_count_cache: Dict[str, int] = {}

    def list_open_markets(
        self,
        limit: int = 1000,
        min_time_to_close_sec: int = 1800,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        out: List[Dict[str, Any]] = []
        sts = statuses or ["open"]
        for st in sts:
            cursor: Optional[str] = None
            while True:
                resp = self.data_client.list_markets(status=st, limit=limit, cursor=cursor)
                rows = resp.get("markets", []) or []
                for m in rows:
                    close_dt = _to_dt(m.get("close_time") or m.get("close_ts"))
                    if close_dt is not None:
                        ttc = (close_dt - now).total_seconds()
                        if ttc < min_time_to_close_sec:
                            continue
                    out.append(m)
                cursor = resp.get("cursor")
                if not cursor:
                    break
        return out

    def market_metadata(self, m: Dict[str, Any]) -> MarketMetadataFeatures:
        yes = m.get("yes_price")
        no = m.get("no_price")
        yes_bid = m.get("yes_bid")
        no_bid = m.get("no_bid")
        yes_ask = m.get("yes_ask")
        if yes_ask is None and no_bid is not None:
            yes_ask = 100 - no_bid
        mid = None
        if yes_bid is not None and yes_ask is not None:
            mid = (float(yes_bid) + float(yes_ask)) / 2.0
        elif yes is not None:
            mid = float(yes)
        close_dt = _to_dt(m.get("close_time") or m.get("close_ts"))
        now = datetime.now(timezone.utc)
        ttc = (close_dt - now).total_seconds() if close_dt else None
        return MarketMetadataFeatures(
            ticker=str(m.get("ticker", "")),
            event_ticker=str(m.get("event_ticker") or m.get("event_id") or ""),
            status=str(m.get("status", "")),
            yes_price=float(yes) if yes is not None else None,
            no_price=float(no) if no is not None else None,
            mid_price=mid,
            close_time=close_dt.isoformat() if close_dt else None,
            time_to_close_sec=ttc,
            volume_24h=float(m.get("volume_24h", 0) or m.get("volume", 0) or 0),
            last_updated=str(m.get("updated_time") or m.get("last_updated") or "") or None,
        )

    def orderbook_features(self, ticker: str, depth_window_cents: int = 5) -> OrderbookFeatures:
        ob = self.data_client.get_orderbook(ticker)
        yes = ob.get("yes", []) or []
        no = ob.get("no", []) or []
        best_yes_bid = _best_price(yes)
        best_no_bid = _best_price(no)
        best_yes_ask = 100 - best_no_bid if best_no_bid is not None else None
        best_no_ask = 100 - best_yes_bid if best_yes_bid is not None else None
        spread_yes = (best_yes_ask - best_yes_bid) if best_yes_ask is not None and best_yes_bid is not None else None
        spread_no = (best_no_ask - best_no_bid) if best_no_ask is not None and best_no_bid is not None else None
        avg_spread = None
        if spread_yes is not None and spread_no is not None:
            avg_spread = (spread_yes + spread_no) / 2.0
        depth_yes = _depth_within(yes, best_yes_bid, depth_window_cents)
        depth_no = _depth_within(no, best_no_bid, depth_window_cents)
        vwap_yes = _vwap_depth(yes, best_yes_bid, depth_window_cents)
        vwap_no = _vwap_depth(no, best_no_bid, depth_window_cents)
        return OrderbookFeatures(
            best_yes_bid=best_yes_bid,
            best_yes_ask=best_yes_ask,
            best_no_bid=best_no_bid,
            best_no_ask=best_no_ask,
            spread_yes=float(spread_yes) if spread_yes is not None else None,
            spread_no=float(spread_no) if spread_no is not None else None,
            avg_spread=avg_spread,
            depth_yes_within_cents=depth_yes,
            depth_no_within_cents=depth_no,
            vwap_yes_depth=vwap_yes,
            vwap_no_depth=vwap_no,
        )

    def trade_features(self, ticker: str, limit: int = 500) -> TradeFeatures:
        now = datetime.now(timezone.utc)
        resp = self.data_client.get_trades(
            ticker=ticker,
            min_ts=int((now - timedelta(hours=24)).timestamp()),
            limit=limit,
        )
        trades = resp.get("trades", []) or []
        prices: List[float] = []
        c1h = 0
        c24h = 0
        for tr in trades:
            ts = _to_dt(tr.get("ts") or tr.get("timestamp") or tr.get("time") or tr.get("created_time"))
            if ts is None:
                continue
            age = (now - ts).total_seconds()
            if age <= 3600:
                c1h += 1
            if age <= 86400:
                c24h += 1
            px = tr.get("yes_price") or tr.get("price")
            if px is not None:
                prices.append(float(px))
        velocity = (c1h / max(1.0, float(c24h))) if c24h > 0 else 0.0
        avg = sum(prices) / len(prices) if prices else None
        disp = None
        if prices and avg is not None:
            disp = math.sqrt(sum((p - avg) ** 2 for p in prices) / len(prices))
        return TradeFeatures(
            trade_count_1h=c1h,
            trade_count_24h=c24h,
            trade_velocity=velocity,
            avg_trade_price=avg,
            trade_price_dispersion=disp,
        )

    def probability_features(self, ticker: str, yes_price: Optional[float], no_price: Optional[float], alpha: float = 0.2) -> ProbabilityMomentumFeatures:
        p_yes = implied_prob(yes_price)
        p_no = implied_prob(no_price if no_price is not None else (100 - yes_price if yes_price is not None else None))
        prev_yes = self._prev_p_yes.get(ticker)
        prev_no = self._prev_p_no.get(ticker)
        dp_yes = (p_yes - prev_yes) if (p_yes is not None and prev_yes is not None) else None
        dp_no = (p_no - prev_no) if (p_no is not None and prev_no is not None) else None
        ema_yes = None
        ema_no = None
        if p_yes is not None:
            ema_yes = alpha * p_yes + (1.0 - alpha) * self._ema_yes.get(ticker, p_yes)
            self._ema_yes[ticker] = ema_yes
            self._prev_p_yes[ticker] = p_yes
        if p_no is not None:
            ema_no = alpha * p_no + (1.0 - alpha) * self._ema_no.get(ticker, p_no)
            self._ema_no[ticker] = ema_no
            self._prev_p_no[ticker] = p_no
        return ProbabilityMomentumFeatures(
            p_yes=p_yes,
            p_no=p_no,
            dp_yes=dp_yes,
            dp_no=dp_no,
            ema_yes=ema_yes,
            ema_no=ema_no,
        )

    def liquidity_edge_features(
        self,
        trade_features: TradeFeatures,
        market_features: MarketMetadataFeatures,
        book_features: OrderbookFeatures,
        p_model_yes: Optional[float] = None,
        p_model_no: Optional[float] = None,
        a: float = 1.0,
        b: float = 0.5,
        c: float = 1.0,
        cost_terms: float = 0.0,
    ) -> LiquidityEdgeFeatures:
        avg_spread = book_features.avg_spread
        liquidity = None
        if avg_spread is not None:
            liquidity = a * trade_features.trade_count_24h + b * market_features.volume_24h - c * avg_spread
        p_yes = implied_prob(market_features.yes_price if market_features.yes_price is not None else market_features.mid_price)
        p_no = implied_prob(market_features.no_price if market_features.no_price is not None else (100 - market_features.mid_price if market_features.mid_price is not None else None))
        p_gap_yes = (p_model_yes - p_yes) if (p_model_yes is not None and p_yes is not None) else None
        p_gap_no = (p_model_no - p_no) if (p_model_no is not None and p_no is not None) else None
        eu_yes = None
        eu_no = None
        if p_model_yes is not None and market_features.yes_price is not None:
            y = market_features.yes_price / 100.0
            eu_yes = p_model_yes * (1.0 - y) - (1.0 - p_model_yes) * y - cost_terms
        if p_model_no is not None and market_features.no_price is not None:
            n = market_features.no_price / 100.0
            eu_no = p_model_no * (1.0 - n) - (1.0 - p_model_no) * n - cost_terms
        return LiquidityEdgeFeatures(
            liquidity_score=liquidity,
            p_gap_yes=p_gap_yes,
            p_gap_no=p_gap_no,
            eu_yes=eu_yes,
            eu_no=eu_no,
        )

    def consistency_features(self, event_ticker: str, limit: int = 500) -> ConsistencyFeatures:
        if not event_ticker:
            return ConsistencyFeatures(sum_of_p_yes=None, inconsistency=None)
        resp = self.data_client.list_markets(event_ticker=event_ticker, limit=limit)
        rows = resp.get("markets", []) or []
        probs: List[float] = []
        for m in rows:
            yes = m.get("yes_price")
            if yes is None:
                yb = m.get("yes_bid")
                ya = m.get("yes_ask")
                if yb is not None and ya is not None:
                    yes = (float(yb) + float(ya)) / 2.0
            p = implied_prob(yes)
            if p is not None:
                probs.append(p)
        if not probs:
            return ConsistencyFeatures(sum_of_p_yes=None, inconsistency=None)
        s = sum(probs)
        return ConsistencyFeatures(sum_of_p_yes=s, inconsistency=abs(s - 1.0))

    def build_feature_vector(
        self,
        market: Dict[str, Any],
        p_model_yes: Optional[float] = None,
        p_model_no: Optional[float] = None,
        depth_window_cents: int = 5,
        alpha_ema: float = 0.2,
    ) -> KalshiFeatureVector:
        meta = self.market_metadata(market)
        book = self.orderbook_features(meta.ticker, depth_window_cents=depth_window_cents)
        trades = self.trade_features(meta.ticker)
        probs = self.probability_features(meta.ticker, meta.yes_price or meta.mid_price, meta.no_price, alpha=alpha_ema)
        liq = self.liquidity_edge_features(
            trade_features=trades,
            market_features=meta,
            book_features=book,
            p_model_yes=p_model_yes,
            p_model_no=p_model_no,
            cost_terms=(book.avg_spread or 0.0) / 100.0,
        )
        consistency = self.consistency_features(meta.event_ticker)
        return KalshiFeatureVector(
            market=meta,
            orderbook=book,
            trades=trades,
            probability=probs,
            liquidity=liq,
            consistency=consistency,
        )

    async def stream_feature_updates(
        self,
        market_tickers: List[str],
        callback: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        """
        Update internal state from ticker + orderbook websocket messages and emit feature snapshots.
        """
        async for msg in self.ws.stream_ticker_and_book(market_tickers):
            mtype = msg.get("type")
            data = msg.get("msg") or msg.get("data") or msg
            ticker = str(data.get("market_ticker") or data.get("ticker") or "")
            if not ticker:
                continue
            if mtype in ("orderbook_snapshot", "orderbook_delta"):
                state = self._book_cache.setdefault(ticker, {"yes": {}, "no": {}})
                if mtype == "orderbook_snapshot":
                    state["yes"] = {int(p): int(s) for p, s in (data.get("yes") or [])}
                    state["no"] = {int(p): int(s) for p, s in (data.get("no") or [])}
                else:
                    side = str(data.get("side") or "").lower()
                    price = data.get("price")
                    delta = data.get("delta") or data.get("size")
                    if side in {"yes", "no"} and price is not None and delta is not None:
                        mp = state[side]
                        p = int(price)
                        v = int(delta)
                        nv = mp.get(p, 0) + v if "delta" in data else v
                        if nv <= 0:
                            mp.pop(p, None)
                        else:
                            mp[p] = nv
                yes_levels = sorted(state["yes"].items(), key=lambda x: x[0], reverse=True)
                no_levels = sorted(state["no"].items(), key=lambda x: x[0], reverse=True)
                best_yes = yes_levels[0][0] if yes_levels else None
                best_no = no_levels[0][0] if no_levels else None
                best_yes_ask = 100 - best_no if best_no is not None else None
                snapshot = {
                    "ticker": ticker,
                    "yes_bid": best_yes,
                    "no_bid": best_no,
                    "yes_ask": best_yes_ask,
                    "type": mtype,
                }
                res = callback(ticker, snapshot)
                if asyncio.iscoroutine(res):
                    await res
            elif mtype == "ticker":
                yes_price = data.get("yes_price") or data.get("yes")
                no_price = data.get("no_price") or data.get("no")
                probs = self.probability_features(ticker, yes_price, no_price)
                payload = {"ticker": ticker, "type": "ticker", "probability": asdict(probs)}
                res = callback(ticker, payload)
                if asyncio.iscoroutine(res):
                    await res

