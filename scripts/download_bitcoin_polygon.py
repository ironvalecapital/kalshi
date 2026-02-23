#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx

UTC = timezone.utc


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


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


def fetch_aggs(
    client: httpx.Client,
    api_key: str,
    multiplier: int,
    timespan: str,
    start: datetime,
    end: datetime,
    symbol: str = "X:BTCUSD",
) -> List[Dict[str, Any]]:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.date()}/{end.date()}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    r = client.get(url, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    return js.get("results", []) or []


def main() -> None:
    api_key = os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_KEY")
    if not api_key:
        raise SystemExit("Missing POLYGON_API_KEY (or POLYGON_KEY)")

    root = Path("data/bitcoin") / datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_polygon")
    root.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=UTC)
    plans = [
        # label, multiplier, timespan, start
        ("1d", 1, "day", datetime(2015, 1, 1, tzinfo=UTC)),
        ("1h_3y", 1, "hour", now - timedelta(days=365 * 3)),
        ("5m_180d", 5, "minute", now - timedelta(days=180)),
        ("1m_30d", 1, "minute", now - timedelta(days=30)),
    ]

    manifest: Dict[str, Any] = {
        "generated_at": now.isoformat(),
        "root": str(root),
        "source": "polygon",
        "symbol": "X:BTCUSD",
        "files": [],
        "errors": [],
    }

    with httpx.Client(headers={"User-Agent": "kalshi-polygon-btc/1.0"}) as client:
        for label, mult, span, start in plans:
            try:
                rows = fetch_aggs(client, api_key=api_key, multiplier=mult, timespan=span, start=start, end=now)
                path_csv = root / f"polygon_btcusd_{label}.csv"
                n = _write_csv(
                    path_csv,
                    ["ts_ms", "ts_iso", "open", "high", "low", "close", "volume", "vwap", "transactions"],
                    (
                        (
                            int(x.get("t", 0)),
                            _iso(int(x.get("t", 0))),
                            x.get("o"),
                            x.get("h"),
                            x.get("l"),
                            x.get("c"),
                            x.get("v"),
                            x.get("vw"),
                            x.get("n"),
                        )
                        for x in rows
                    ),
                )
                _write_json(root / f"polygon_btcusd_{label}.json", rows)
                manifest["files"].append({"label": label, "rows": n, "csv": str(path_csv)})
                time.sleep(0.25)
            except Exception as exc:
                manifest["errors"].append({"label": label, "error": str(exc)})

    manifest["total_rows"] = int(sum(x.get("rows", 0) for x in manifest["files"]))
    _write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
