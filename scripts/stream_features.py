#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict

import numpy as np

# Ensure project root is importable when run as `python scripts/stream_features.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalshi_engine.pipeline import build_market_decision, decision_to_dict
from kalshi_engine.sports_bayes import SportsGameState
from kalshi_engine.storage import ParquetStore
from kalshi_engine.kalshi_ws import KalshiWSClient


def _mock_sports_state(implied_prob: float) -> SportsGameState:
    return SportsGameState(
        score_diff=0.0,
        time_remaining_min=24.0,
        has_possession=True,
        timeouts_team=3,
        timeouts_opp=3,
        pregame_prob=max(0.01, min(0.99, implied_prob)),
        historical_scoring_rate_per_min=2.1,
        current_kalshi_implied=max(0.01, min(0.99, implied_prob)),
    )


async def run(args: argparse.Namespace) -> None:
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided")

    ws = KalshiWSClient(demo=args.demo)
    store = ParquetStore(root_dir=args.out)
    btc_returns = np.asarray([0.0] * 120, dtype=float)
    last_price: Dict[str, float] = {}

    task = asyncio.create_task(ws.connect_and_stream(tickers=tickers, channels=["orderbook_delta", "trade", "ticker_v2"]))
    try:
        while True:
            now = datetime.now(timezone.utc).isoformat()
            for ticker in tickers:
                book = ws.books.get(ticker)
                if not book:
                    continue
                yes_levels = book.sorted_yes()
                no_levels = book.sorted_no()
                trades = list(ws.trades.get(ticker, []))
                implied_prob = 0.5
                if yes_levels and no_levels:
                    implied_prob = (yes_levels[0][0] + (100 - no_levels[0][0])) / 200.0

                if trades:
                    p = trades[-1].get("yes_price")
                    if p is not None:
                        pr = float(p) / 100.0
                        if ticker in last_price:
                            btc_returns = np.append(btc_returns[-500:], pr - last_price[ticker])
                        last_price[ticker] = pr

                sports_state = _mock_sports_state(implied_prob) if args.use_sports else None
                decision = build_market_decision(
                    market_ticker=ticker,
                    yes_levels=yes_levels,
                    no_levels=no_levels,
                    trades=trades,
                    bankroll=args.bankroll,
                    target_order_size=args.target_size,
                    event_ticker=None,
                    close_time=None,
                    status="open",
                    model_prob_hint=implied_prob,
                    sports_state=sports_state,
                    btc_returns=btc_returns if args.use_btc_regime else None,
                )
                payload = decision_to_dict(decision)
                payload["ts"] = now
                print(json.dumps(payload, default=str))
                store.write_features([
                    {
                        "market_ticker": ticker,
                        "ts": now,
                        "mode": payload.get("mode"),
                        "model_prob": payload.get("snapshot", {}).get("model", {}).get("model_prob"),
                        "implied_prob": payload.get("snapshot", {}).get("liquidity", {}).get("implied_prob"),
                        "edge": payload.get("snapshot", {}).get("model", {}).get("edge"),
                        "emotion_spike_score": payload.get("snapshot", {}).get("flow", {}).get("emotion_spike_score"),
                        "btc_tilt_score": payload.get("snapshot", {}).get("flow", {}).get("btc_tilt_score"),
                        "raw_json": payload,
                    }
                ])
            await asyncio.sleep(args.interval)
    finally:
        task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Kalshi L2+trade features and print LLM snapshots")
    parser.add_argument("--tickers", required=True, help="Comma-separated market tickers")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between feature snapshots")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Bankroll used for sizing context")
    parser.add_argument("--target-size", type=int, default=10, help="Target order size for slippage estimate")
    parser.add_argument("--out", default="data/kalshi_parquet", help="Parquet root output")
    parser.add_argument("--demo", action="store_true", help="Use demo websocket endpoint")
    parser.add_argument("--use-sports", action="store_true", help="Enable Sports Bayesian posterior in snapshot")
    parser.add_argument("--use-btc-regime", action="store_true", help="Enable BTC regime classification in snapshot")
    ns = parser.parse_args()
    asyncio.run(run(ns))
