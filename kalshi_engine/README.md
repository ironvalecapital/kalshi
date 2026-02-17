# Kalshi Engine Data Layer

Async ingestion + feature layer for LLM-assisted Kalshi trading.

## Modules
- `kalshi_rest.py`: async REST client with retry and basic 429 handling.
- `kalshi_ws.py`: websocket stream client, local L2 orderbook state (snapshot + delta), rolling trade tape.
- `models.py`: typed payload models (`pydantic`).
- `features.py`: compact market snapshot features for LLM/risk engines.
- `sports_bayes.py`: Bayesian sports win-probability posterior.
- `btc_regime.py`: BTC volatility regime detector (HMM + score-based calm/expansion/panic classifier).
- `internal_arb.py`: within-Kalshi structural opportunity checks (complementary yes/no + event basket consistency).
- `pipeline.py`: orchestrates sports + btc + internal-arb into one decision payload.
- `storage.py`: parquet partition sink (`date=.../market_ticker=...`).
- `backfill.py`: historical pull for trades/candles into parquet.

## WebSocket Subscribe Shape
Kalshi subscribe payload used by `kalshi_ws.py`:

```json
{
  "id": 1739900000000,
  "cmd": "subscribe",
  "params": {
    "channels": ["orderbook_delta", "trade", "ticker_v2"],
    "market_tickers": ["KXBTC15M-26FEB141230-30"]
  }
}
```

## Runner
Stream snapshots every N seconds:

```bash
python3 scripts/stream_features.py \
  --tickers KXBTC15M-26FEB141230-30,KXETH15M-26FEB141230-30 \
  --interval 5 \
  --use-btc-regime \
  --use-sports
```

## Backfill

```bash
python3 -m kalshi_engine.backfill --tickers KXBTC15M-26FEB141230-30 --out data/kalshi_parquet
```

## Notes
- Public endpoints/channels work without signed auth. Add your signing wrapper if you need private channels.
- Feature snapshots are intentionally compact and stable for LLM prompts.
