# Kalshi Decision Engine

SAFE, compliant Kalshi algorithmic trading bot with conservative decisioning, strict risk limits, and full auditability. Demo mode is the default.

## Features
- Official Kalshi REST + WebSocket integration (no scraping).
- Three decision modules: Bayesian prior tracking, cross-market consistency checks, and event-time microstructure.
- Strict risk management: daily loss caps, per-market limits, max open orders, max notional, cooldowns.
- Rate limit guardrails by tier with token-bucket throttling.
- Order management with maker preference and kill switch.
- SQLite ledger and JSON audit trail for every decision.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
- Copy `.env.example` and fill in `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`.
- Edit `configs/example.yaml` for risk limits, rate limit tier, and execution behavior.

## Demo First
```bash
kalshi-bot doctor
kalshi-bot markets --query "inflation" --top 5
kalshi-bot run --strategy bayes --market CPI-24APR --demo --once
```

## Weather Market Picker (NWS)
```bash
kalshi-bot pick-weather --city NYC
```

## Weather Lane Quickstart
```bash
# Demo-only run (safe)
kalshi-bot run-weather --demo --cycles 2 --sleep 2
```

Safety notes:
- Live trading requires `--live --i-understand-risk`.
- Kill switch: `KALSHI_BOT_KILL=1`.

## Live Trading (Explicit)
```bash
kalshi-bot run --strategy bayes --market CPI-24APR --live --i-understand-risk
```

## Safety
- Kill switch: set `KALSHI_BOT_KILL=1` to stop trading immediately.
- If rate limits (429) or auth errors (403) spike, the bot pauses and backs off.
- Minimum order interval enforced to avoid spam or quote stuffing.

## Notes
- `kalshi-python` is used where possible, but current Kalshi docs recommend `kalshi-python-sync` for newer endpoints. This project keeps compatibility with `kalshi-python` and relies on direct REST calls for endpoints not exposed in that SDK.

## Helpful Links
- Kalshi API docs: [docs.kalshi.com](https://docs.kalshi.com/)
- Kalshi rate limits: [Rate Limits](https://docs.kalshi.com/getting_started/rate_limits)
- Kalshi orderbook endpoint: [Get Market Orderbook](https://docs.kalshi.com/api-reference/markets/get-market-orderbook)
- Kalshi trades endpoint: [Get Trades](https://docs.kalshi.com/api-reference/markets/get-trades)
- Kalshi order create: [Create Order](https://docs.kalshi.com/api-reference/orders/create-order)
- Kalshi fee schedule (PDF): [Fee Schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
- Kalshi authenticated requests: [Quick Start Auth](https://docs.kalshi.com/getting_started/quick_start_authenticated_requests)
- NWS API (User-Agent required): [api.weather.gov](https://api.weather.gov/)
- NWS points endpoint: [Points API](https://www.weather.gov/documentation/services-web-api#/default/point)
- Render deployment docs: [Render Python](https://render.com/docs/deploy-python)

## Files
- `kalshi_bot/` core bot modules
- `docs/OPERATING_RULES.md` safe operation and incident procedures
