#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalshi_engine.kalshi_rest import KalshiRESTClient
from kalshi_bot.squiggle_export import is_focus_sports_market


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def fetch_markets(client: KalshiRESTClient, statuses: List[str], max_pages: int, per_page: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for status in statuses:
        cursor = None
        page = 0
        while page < max_pages:
            resp = await client.list_markets(status=status, limit=per_page, cursor=cursor)
            rows = resp.get("markets", []) or []
            if not rows:
                break
            out.extend(rows)
            cursor = resp.get("cursor")
            page += 1
            if not cursor:
                break
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Download Kalshi sports/UFC/basketball focus datasets")
    ap.add_argument("--out", default="data/sports_focus", help="output directory")
    ap.add_argument("--max-pages", type=int, default=40, help="max pages per status")
    ap.add_argument("--per-page", type=int, default=200, help="markets per page")
    ap.add_argument("--trades-limit", type=int, default=1200, help="recent trades per market")
    ap.add_argument("--statuses", default="open,unopened", help="comma-separated statuses")
    ap.add_argument("--download-orderbooks", action="store_true", help="download one L2 snapshot per market")
    args = ap.parse_args()

    out_root = Path(args.out)
    ts = _now_tag()
    run_dir = out_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    statuses = [s.strip() for s in args.statuses.split(",") if s.strip()]
    client = KalshiRESTClient()
    try:
        all_markets = await fetch_markets(client, statuses=statuses, max_pages=args.max_pages, per_page=args.per_page)
        focus = [m for m in all_markets if is_focus_sports_market(m)]

        (run_dir / "markets_all.json").write_text(json.dumps(all_markets, indent=2), encoding="utf-8")
        (run_dir / "markets_focus.json").write_text(json.dumps(focus, indent=2), encoding="utf-8")

        trades_dir = run_dir / "trades"
        orderbooks_dir = run_dir / "orderbooks"
        trades_dir.mkdir(parents=True, exist_ok=True)
        if args.download_orderbooks:
            orderbooks_dir.mkdir(parents=True, exist_ok=True)

        summary: List[Dict[str, Any]] = []
        for i, m in enumerate(focus, start=1):
            ticker = str(m.get("ticker") or "").strip()
            if not ticker:
                continue
            row: Dict[str, Any] = {
                "ticker": ticker,
                "title": m.get("title"),
                "status": m.get("status"),
                "volume_24h": m.get("volume_24h"),
                "yes_bid": m.get("yes_bid"),
                "no_bid": m.get("no_bid"),
                "trades_downloaded": 0,
                "orderbook_downloaded": False,
            }
            try:
                tr = await client.get_trades(ticker=ticker, limit=args.trades_limit)
                trades = tr.get("trades", []) or []
                (trades_dir / f"{ticker}.json").write_text(json.dumps(trades, indent=2), encoding="utf-8")
                row["trades_downloaded"] = len(trades)
            except Exception as exc:
                row["trades_error"] = str(exc)

            if args.download_orderbooks:
                try:
                    ob = await client.get_orderbook(ticker)
                    (orderbooks_dir / f"{ticker}.json").write_text(json.dumps(ob, indent=2), encoding="utf-8")
                    row["orderbook_downloaded"] = True
                except Exception as exc:
                    row["orderbook_error"] = str(exc)

            summary.append(row)
            if i % 25 == 0:
                print(f"downloaded {i}/{len(focus)} focus markets")

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        counts = {
            "run_dir": str(run_dir),
            "markets_all": len(all_markets),
            "markets_focus": len(focus),
            "trades_files": len(list(trades_dir.glob("*.json"))),
            "orderbooks_files": len(list(orderbooks_dir.glob("*.json"))) if args.download_orderbooks else 0,
        }
        print(json.dumps(counts, indent=2))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
