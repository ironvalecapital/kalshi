# Operating Rules

## Core Principles
- Demo mode is default. Live trading requires `--live --i-understand-risk`.
- Never run without risk limits enabled.
- No manipulative behavior: no spam orders, no quote stuffing, no rapid cancel/replace loops.

## Risk Controls
- Daily loss limits, max position per market, max open orders, and max notional enforced every order.
- Cooldown enforced after consecutive losses.
- Minimum order interval enforced per market.
- Kill switch: set `KALSHI_BOT_KILL=1` to halt trading immediately.

## Rate Limits
- Token-bucket limiter enforces per-tier read/write rates.
- If 429/403 errors spike, trading pauses and backs off.

## Incident Procedures
1. Set `KALSHI_BOT_KILL=1`.
2. Review `logs/audit.log` and the SQLite ledger `kalshi.db`.
3. Validate open orders and positions via `kalshi-bot doctor`.
4. Restart only after root cause identified and risk limits reviewed.

## Data Integrity
- Every decision is written to both the audit log and SQLite ledger.
- Decision reports are emitted to `runs/decision_report_*.json` for each cycle.
