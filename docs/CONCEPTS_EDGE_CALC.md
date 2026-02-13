# Concepts: Edge Calculation

This document explains the core math used by the market scanner and edge scoring modules.

**LiquidityScore**

We use a weighted liquidity score based on volume and trade rate with a spread penalty:

```
LiquidityScore = α * volume_24h + β * trades_1h − γ * spread
```

**TradabilityScore**

We extend liquidity with a short-term activity term:

```
TradabilityScore = LiquidityScore + 0.2 * trades_24h − 0.5 * spread
```

**Implied Probability**

For a binary contract, the implied probability is:

```
P_implied = price_cents / 100
```

**Expected Utility (EV)**

For BUY YES:

```
EV = p_model*(1 − price) − (1 − p_model)*price − spread − fee − slippage
```

For BUY NO:

```
EV = (1 − p_model)*(1 − price) − p_model*price − spread − fee − slippage
```

Where `price` is in dollars (cents / 100). In code, EV is tracked in cents per contract.

**Maker Laddering**

The maker ladder places post-only limit orders at multiple levels inside the spread:

```
price_level_i = best_bid + 1 + i
```

Orders are only placed when `EV >= min_ev_cents` and risk constraints permit.

**Risk Constraints**

- `max_daily_loss_usd`
- `max_position_per_market_usd`
- `max_open_orders`
- `min_order_interval_seconds`

These hard caps ensure the engine does not trade when risk limits are exceeded.
