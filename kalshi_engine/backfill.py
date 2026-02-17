from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .kalshi_rest import KalshiRESTClient
from .storage import ParquetStore


async def backfill_trades(
    client: KalshiRESTClient,
    store: ParquetStore,
    ticker: str,
    limit: int = 1000,
) -> int:
    data = await client.get_trades(ticker=ticker, limit=limit)
    rows: List[Dict[str, Any]] = []
    for tr in data.get("trades", []) or []:
        rows.append(
            {
                "market_ticker": tr.get("ticker") or ticker,
                "ts": tr.get("created_time") or tr.get("ts") or datetime.now(timezone.utc).isoformat(),
                "yes_price": tr.get("yes_price") or tr.get("price"),
                "no_price": tr.get("no_price"),
                "count": tr.get("count") or 0,
                "raw_json": tr,
            }
        )
    store.write_trades(rows)
    return len(rows)


async def backfill_candles(
    client: KalshiRESTClient,
    store: ParquetStore,
    ticker: str,
    period_interval: int = 1,
    period_unit: str = "minute",
) -> int:
    try:
        data = await client.get_candlesticks(ticker=ticker, period_interval=period_interval, period_unit=period_unit)
    except Exception:
        return 0
    rows: List[Dict[str, Any]] = []
    for c in data.get("candlesticks", []) or data.get("candles", []) or []:
        rows.append(
            {
                "market_ticker": ticker,
                "ts": c.get("end_period_ts") or c.get("ts") or c.get("time"),
                "open": c.get("open"),
                "high": c.get("high"),
                "low": c.get("low"),
                "close": c.get("close"),
                "volume": c.get("volume") or 0.0,
                "raw_json": c,
            }
        )
    store.write_ticks(rows)
    return len(rows)


async def run_backfill(
    tickers: list[str],
    root_dir: str = "data/kalshi_parquet",
) -> Dict[str, int]:
    client = KalshiRESTClient()
    store = ParquetStore(root_dir=root_dir)
    out: Dict[str, int] = {}
    try:
        for t in tickers:
            n_tr = await backfill_trades(client, store, t, limit=2000)
            n_cd = await backfill_candles(client, store, t, period_interval=1, period_unit="minute")
            out[t] = n_tr + n_cd
    finally:
        await client.close()
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tickers", required=True, help="Comma-separated market tickers")
    p.add_argument("--out", default="data/kalshi_parquet", help="Parquet output root")
    args = p.parse_args()
    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    stats = asyncio.run(run_backfill(tickers=tickers, root_dir=args.out))
    print(stats)
