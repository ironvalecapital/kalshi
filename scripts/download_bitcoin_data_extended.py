#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx

UTC = timezone.utc


def _iso_s(ts_s: int) -> str:
    return datetime.fromtimestamp(ts_s, tz=UTC).isoformat()


def _iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


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


def fetch_yahoo(client: httpx.Client, out_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for interval, rng in (("1d", "max"), ("1h", "730d"), ("5m", "60d")):
        url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD"
        r = client.get(url, params={"interval": interval, "range": rng, "events": "history"}, timeout=60)
        r.raise_for_status()
        js = r.json()
        _write_json(out_dir / f"yahoo_BTC-USD_{interval}_{rng}.json", js)
        result = js.get("chart", {}).get("result", [{}])[0]
        ts = result.get("timestamp", []) or []
        q = (result.get("indicators", {}).get("quote", [{}]) or [{}])[0]
        rows = _write_csv(
            out_dir / f"yahoo_BTC-USD_{interval}_{rng}.csv",
            ["ts_sec", "ts_iso", "open", "high", "low", "close", "volume"],
            (
                (
                    int(ts[i]),
                    _iso_s(int(ts[i])),
                    q.get("open", [None])[i],
                    q.get("high", [None])[i],
                    q.get("low", [None])[i],
                    q.get("close", [None])[i],
                    q.get("volume", [None])[i],
                )
                for i in range(len(ts))
            ),
        )
        out[f"yahoo_{interval}"] = rows
    return out


def fetch_coinbase_candles(client: httpx.Client, out_dir: Path, granularity: int, start: datetime, end: datetime) -> int:
    # Coinbase returns max 300 candles per call.
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    cursor = start
    all_rows: List[List[Any]] = []
    chunk = timedelta(seconds=granularity * 300)
    while cursor < end:
        to = min(end, cursor + chunk)
        r = client.get(
            url,
            params={
                "granularity": granularity,
                "start": cursor.isoformat().replace("+00:00", "Z"),
                "end": to.isoformat().replace("+00:00", "Z"),
            },
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json() or []
        all_rows.extend(batch)
        cursor = to
        time.sleep(0.06)

    # Coinbase candles: [time, low, high, open, close, volume]
    seen = set()
    uniq = []
    for r in all_rows:
        t = int(r[0])
        if t in seen:
            continue
        seen.add(t)
        uniq.append(r)
    uniq.sort(key=lambda x: x[0])

    name = f"coinbase_BTC-USD_{granularity}s"
    rows = _write_csv(
        out_dir / f"{name}.csv",
        ["ts_sec", "ts_iso", "open", "high", "low", "close", "volume"],
        ((int(x[0]), _iso_s(int(x[0])), float(x[3]), float(x[2]), float(x[1]), float(x[4]), float(x[5])) for x in uniq),
    )
    _write_json(out_dir / f"{name}_meta.json", {"rows": rows, "start": start.isoformat(), "end": end.isoformat()})
    return rows


def fetch_kraken(client: httpx.Client, out_dir: Path, interval: int, since_s: int) -> int:
    url = "https://api.kraken.com/0/public/OHLC"
    all_rows: List[List[Any]] = []
    while True:
        r = client.get(url, params={"pair": "XBTUSD", "interval": interval, "since": since_s}, timeout=60)
        r.raise_for_status()
        js = r.json()
        rows = js.get("result", {}).get("XXBTZUSD", []) or js.get("result", {}).get("XBTUSD", []) or []
        last = int(js.get("result", {}).get("last", since_s))
        if not rows:
            break
        all_rows.extend(rows)
        if last <= since_s:
            break
        since_s = last
        time.sleep(0.08)
        if len(rows) < 700:
            break

    seen = set()
    uniq = []
    for r in all_rows:
        t = int(float(r[0]))
        if t in seen:
            continue
        seen.add(t)
        uniq.append(r)
    uniq.sort(key=lambda x: float(x[0]))

    name = f"kraken_XBTUSD_{interval}m"
    rows = _write_csv(
        out_dir / f"{name}.csv",
        ["ts_sec", "ts_iso", "open", "high", "low", "close", "vwap", "volume", "count"],
        (
            (
                int(float(x[0])),
                _iso_s(int(float(x[0]))),
                float(x[1]),
                float(x[2]),
                float(x[3]),
                float(x[4]),
                float(x[5]),
                float(x[6]),
                int(float(x[7])),
            )
            for x in uniq
        ),
    )
    return rows


def main() -> None:
    root = Path("data/bitcoin") / datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"root": str(root), "sources": []}
    with httpx.Client(headers={"User-Agent": "kalshi-bitcoin-downloader/1.0"}) as client:
        # Yahoo
        try:
            stats = fetch_yahoo(client, root)
            for k, v in stats.items():
                manifest["sources"].append({"name": k, "rows": v})
        except Exception as exc:
            _write_json(root / "yahoo_error.json", {"error": str(exc)})

        # Coinbase
        try:
            now = datetime.now(tz=UTC)
            rows = fetch_coinbase_candles(client, root, granularity=86400, start=datetime(2015, 1, 1, tzinfo=UTC), end=now)
            manifest["sources"].append({"name": "coinbase_1d", "rows": rows})
        except Exception as exc:
            _write_json(root / "coinbase_1d_error.json", {"error": str(exc)})

        try:
            now = datetime.now(tz=UTC)
            rows = fetch_coinbase_candles(client, root, granularity=3600, start=now - timedelta(days=730), end=now)
            manifest["sources"].append({"name": "coinbase_1h_730d", "rows": rows})
        except Exception as exc:
            _write_json(root / "coinbase_1h_error.json", {"error": str(exc)})

        try:
            now = datetime.now(tz=UTC)
            rows = fetch_coinbase_candles(client, root, granularity=300, start=now - timedelta(days=60), end=now)
            manifest["sources"].append({"name": "coinbase_5m_60d", "rows": rows})
        except Exception as exc:
            _write_json(root / "coinbase_5m_error.json", {"error": str(exc)})

        # Kraken fallback
        try:
            rows = fetch_kraken(client, root, interval=60, since_s=int(datetime(2017, 1, 1, tzinfo=UTC).timestamp()))
            manifest["sources"].append({"name": "kraken_60m", "rows": rows})
        except Exception as exc:
            _write_json(root / "kraken_error.json", {"error": str(exc)})

    manifest["total_rows"] = int(sum(int(x.get("rows", 0)) for x in manifest["sources"]))
    manifest["generated_at"] = datetime.now(tz=UTC).isoformat()
    _write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
