from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..data_rest import KalshiDataClient
from ..ledger import Ledger


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        val = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            # Fall back to stripping fractional seconds beyond 6 digits
            if "." in val:
                base, rest = val.split(".", 1)
                frac = "".join(ch for ch in rest if ch.isdigit())[:6]
                tz = "+00:00" if "+" not in rest else "+" + rest.split("+")[-1]
                return datetime.fromisoformat(f"{base}.{frac}{tz}")
    return datetime.now(timezone.utc)


@dataclass
class TradeRecord:
    trade_id: str
    market_id: str
    ts: datetime
    price: int
    count: int
    raw: Dict[str, Any]


def fetch_trades_paged(
    data_client: KalshiDataClient,
    ticker: str,
    limit: int = 200,
    max_pages: int = 20,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
) -> List[TradeRecord]:
    cursor: Optional[str] = None
    trades: List[TradeRecord] = []
    pages = 0
    while True:
        resp = data_client.get_trades(
            ticker=ticker,
            min_ts=min_ts,
            max_ts=max_ts,
            limit=limit,
            cursor=cursor,
        )
        for tr in resp.get("trades", []):
            ts_val = tr.get("created_time") or tr.get("ts") or tr.get("timestamp") or tr.get("time")
            price = tr.get("yes_price") or tr.get("price") or tr.get("yes_price_cents") or 0
            trades.append(
                TradeRecord(
                    trade_id=str(tr.get("trade_id") or ""),
                    market_id=str(tr.get("ticker") or ticker),
                    ts=_parse_ts(ts_val),
                    price=int(price),
                    count=int(tr.get("count") or 0),
                    raw=tr,
                )
            )
        cursor = resp.get("cursor")
        pages += 1
        if not cursor or pages >= max_pages:
            break
    return trades


def store_trades(ledger: Ledger, trades: List[TradeRecord]) -> int:
    stored = 0
    for tr in trades:
        ledger.record_market_trade(
            market_id=tr.market_id,
            ts=tr.ts.astimezone(timezone.utc).isoformat(),
            price=tr.price,
            count=tr.count,
            raw=tr.raw,
        )
        stored += 1
    return stored


def sync_trades(
    data_client: KalshiDataClient,
    ledger: Ledger,
    ticker: str,
    lookback_sec: int = 3600,
    limit: int = 200,
    max_pages: int = 20,
) -> Tuple[int, int]:
    now = int(datetime.now(tz=timezone.utc).timestamp())
    min_ts = now - lookback_sec
    trades = fetch_trades_paged(
        data_client,
        ticker=ticker,
        limit=limit,
        max_pages=max_pages,
        min_ts=min_ts,
    )
    stored = store_trades(ledger, trades)
    return stored, len(trades)
