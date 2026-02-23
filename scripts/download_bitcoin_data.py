#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx


UTC = timezone.utc


@dataclass
class SourceResult:
    name: str
    path: str
    rows: int


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, headers: List[str], rows: Iterable[Iterable[Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(list(r))
            n += 1
    return n


def fetch_coingecko(client: httpx.Client, out_dir: Path) -> List[SourceResult]:
    out: List[SourceResult] = []
    base = "https://api.coingecko.com/api/v3/coins/bitcoin"

    mkt = client.get(f"{base}/market_chart", params={"vs_currency": "usd", "days": "max"}, timeout=60)
    mkt.raise_for_status()
    mkt_json = mkt.json()
    _write_json(out_dir / "coingecko_market_chart_max.json", mkt_json)
    prices = mkt_json.get("prices", [])
    rows = _write_csv(
        out_dir / "coingecko_market_chart_max_prices.csv",
        ["ts_ms", "ts_iso", "price_usd"],
        ((int(x[0]), _iso(int(x[0])), float(x[1])) for x in prices),
    )
    out.append(SourceResult("coingecko_market_chart_max", str(out_dir / "coingecko_market_chart_max_prices.csv"), rows))

    ohlc = client.get(f"{base}/ohlc", params={"vs_currency": "usd", "days": "max"}, timeout=60)
    if ohlc.status_code == 200:
        ohlc_json = ohlc.json()
        _write_json(out_dir / "coingecko_ohlc_max.json", ohlc_json)
        rows = _write_csv(
            out_dir / "coingecko_ohlc_max.csv",
            ["ts_ms", "ts_iso", "open", "high", "low", "close"],
            ((int(x[0]), _iso(int(x[0])), float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in ohlc_json),
        )
        out.append(SourceResult("coingecko_ohlc_max", str(out_dir / "coingecko_ohlc_max.csv"), rows))

    return out


def fetch_binance_klines(client: httpx.Client, out_dir: Path, interval: str, start_ms: int, end_ms: Optional[int] = None) -> SourceResult:
    url = "https://api.binance.com/api/v3/klines"
    params: Dict[str, Any] = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": 1000,
        "startTime": start_ms,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    data_rows: List[List[Any]] = []
    cursor = start_ms
    safety = 0

    while True:
        params["startTime"] = cursor
        r = client.get(url, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        data_rows.extend(batch)
        last_open = int(batch[-1][0])
        cursor = last_open + 1
        safety += 1
        if len(batch) < 1000:
            break
        if end_ms is not None and last_open >= end_ms:
            break
        if safety > 20000:
            break
        time.sleep(0.08)

    path = out_dir / f"binance_BTCUSDT_{interval}.csv"
    rows = _write_csv(
        path,
        [
            "open_time_ms",
            "open_time_iso",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ],
        (
            (
                int(x[0]),
                _iso(int(x[0])),
                float(x[1]),
                float(x[2]),
                float(x[3]),
                float(x[4]),
                float(x[5]),
                int(x[6]),
                float(x[7]),
                int(x[8]),
                float(x[9]),
                float(x[10]),
            )
            for x in data_rows
        ),
    )
    return SourceResult(f"binance_{interval}", str(path), rows)


def fetch_fear_greed(client: httpx.Client, out_dir: Path) -> SourceResult:
    url = "https://api.alternative.me/fng/"
    r = client.get(url, params={"limit": 0, "format": "json"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    _write_json(out_dir / "alternative_fear_greed.json", js)
    rows = _write_csv(
        out_dir / "alternative_fear_greed.csv",
        ["timestamp", "value", "value_classification", "time_until_update"],
        (
            (
                int(x.get("timestamp", 0)),
                int(x.get("value", 0)),
                x.get("value_classification", ""),
                x.get("time_until_update", ""),
            )
            for x in js.get("data", [])
        ),
    )
    return SourceResult("fear_greed", str(out_dir / "alternative_fear_greed.csv"), rows)


def fetch_blockchain_market_price(client: httpx.Client, out_dir: Path) -> SourceResult:
    # Daily BTCUSD from blockchain.com
    url = "https://api.blockchain.info/charts/market-price"
    r = client.get(url, params={"timespan": "all", "format": "json"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    _write_json(out_dir / "blockchain_market_price_all.json", js)
    vals = js.get("values", [])
    rows = _write_csv(
        out_dir / "blockchain_market_price_all.csv",
        ["ts_sec", "ts_iso", "price_usd"],
        (
            (
                int(x.get("x", 0)),
                datetime.fromtimestamp(int(x.get("x", 0)), tz=UTC).isoformat(),
                float(x.get("y", 0.0)),
            )
            for x in vals
        ),
    )
    return SourceResult("blockchain_market_price", str(out_dir / "blockchain_market_price_all.csv"), rows)


def main() -> None:
    root = Path("data/bitcoin") / datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=UTC)
    start_1d = int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    start_1h = int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    start_5m = int((now - timedelta(days=365)).timestamp() * 1000)
    start_1m = int((now - timedelta(days=90)).timestamp() * 1000)

    results: List[SourceResult] = []
    with httpx.Client(headers={"User-Agent": "kalshi-decision-engine-btc-downloader/1.0"}) as client:
        try:
            results.extend(fetch_coingecko(client, root))
        except Exception as exc:
            _write_json(root / "coingecko_error.json", {"error": str(exc)})

        for interval, start_ms in (("1d", start_1d), ("1h", start_1h), ("5m", start_5m), ("1m", start_1m)):
            try:
                results.append(fetch_binance_klines(client, root, interval=interval, start_ms=start_ms))
            except Exception as exc:
                _write_json(root / f"binance_{interval}_error.json", {"error": str(exc)})

        try:
            results.append(fetch_fear_greed(client, root))
        except Exception as exc:
            _write_json(root / "fear_greed_error.json", {"error": str(exc)})

        try:
            results.append(fetch_blockchain_market_price(client, root))
        except Exception as exc:
            _write_json(root / "blockchain_market_price_error.json", {"error": str(exc)})

    manifest = {
        "generated_at": now.isoformat(),
        "root": str(root),
        "sources": [r.__dict__ for r in results],
        "total_rows": int(sum(r.rows for r in results)),
    }
    _write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
