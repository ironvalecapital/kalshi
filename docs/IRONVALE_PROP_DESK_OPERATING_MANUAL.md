# Ironvale Prop Desk Operating Manual

## 90-Day Build Plan

### Days 1-30 (Core Engine + Logging)
- Run websocket-first ingestion (`orderbook_delta`, `trade`).
- Produce feature snapshots (imbalance, OFI, microprice, spread, depth).
- Use blended probability:
  - `P_model = P_fund + lambda * dP_flow`
- Enforce fee-aware EV gate and fractional Kelly.
- Output full trade journal and Brier score series.

### Days 31-60 (Validation + Risk)
- Walk-forward slices (train/test rolling windows).
- Monte Carlo ruin analysis (10k paths).
- One-sided edge significance test (`H0: mu <= 0`).
- Drawdown governor and per-vertical allocator.

### Days 61-90 (Execution Intelligence)
- Maker/taker switching and toxicity checks.
- Slippage and adverse-selection attribution.
- Simulation-only execution policy optimization.

## Daily SOP

### Pre-session
- Check drawdown, exposure, calibration drift, volatility regime.
- Apply auto size reduction if drawdown exceeds threshold.

### Live session
- Re-score every cycle from fresh market + flow state.
- Only trade if:
  - `edge_after_fees > threshold`
  - liquidity gate passes
  - risk gate passes

### Post-session
- Store journal, update Brier + Sharpe, record anomalies.
- No parameter changes during drawdown windows.

## Weekly/Monthly Review Framework

### Weekly
- Edge net fees vs baseline.
- Slippage and adverse-selection rate.
- Vertical return and exposure decomposition.

### Monthly
- Calibration reliability curve.
- Walk-forward stability report.
- Monte Carlo ruin update.
- Formal parameter review window.

## Hard Controls
- Fractional Kelly cap <= 15% of raw Kelly.
- Per-market exposure <= 5% bankroll.
- Per-vertical exposure <= 40% bankroll.
- Drawdown > 20% => cut sizing multiplier.
- Strategy disable if significance fails over review window.

## Non-Negotiables
- No optimization without out-of-sample validation.
- No parameter change outside review cycle.
- Survival takes priority over growth.

