You are Codex acting as a senior quant engineer. Research how to programmatically fetch Kalshi market data, discover tradable markets with potential edge, and automate order placement through Kalshi’s official APIs. Use official documentation plus real examples of Kalshi trading bots and market-making projects to write code for a production-ready Kalshi algorithmic trading strategy.

RESEARCH SOURCES
- Kalshi official API docs: market data, orderbook, trades, create orders. :contentReference[oaicite:1]{index=1}
- Kalshi API Quick Start: market discovery + order placement. :contentReference[oaicite:2]{index=2}
- WebSocket real-time data feed for orderbooks/trades. :contentReference[oaicite:3]{index=3}
- Example Kalshi bots on GitHub for inspiration. :contentReference[oaicite:4]{index=4}

YOUR TASK:
1) **Market discovery + scanning**
   - Build modules to:
     - Fetch all open Kalshi markets (paginated) via `GET /markets`.
     - Filter markets by status, volume, spread, time-to-close.
     - Incorporate a **spread scanner** (like Polymarket’s) that ranks markets by tightness and depth.
     - Skip markets that are near resolution (e.g., < X minutes to close or priced near 0¢/100¢).

2) **Real-time market data ingest**
   - Implement REST + WebSocket ingestion:
     - REST: initial snapshots (markets, orderbook, trades). :contentReference[oaicite:5]{index=5}
     - WebSockets: subscribe to orderbook deltas, trades, market status, and fills. :contentReference[oaicite:6]{index=6}
   - Maintain a normalized market state:
     - best bid, derived ask, spread, depth, recent trades.

3) **Liquidity and edge heuristics**
   - Compute for each market:
     - **Liquidity score** = f(trade count, spread, volume).
     - **Spread filter**: only trade if spread < threshold.
     - **Depth ladder**: quote multiple layers of passive orders proportional to depth.
     - **Recently active filter**: require minimum trade activity in last N minutes.

4) **Expected value gating**
   - Compute EV after costs for each potential trade:
     - `EV = P(model win) * payout − (spread + fees + slippage)`
   - Algorithm should use model probabilities or simple heuristics (e.g., orderbook imbalance, trend) to estimate edge vs implied prices.

5) **Order management**
   - Implement maker-first default:
     - Place post-only limit orders inside the spread.
     - Cancel stale orders when edge decays or price moves.
   - Use standard parameters:
     - limit price, post only, time in force, reduce only flags.
   - Respect Kalshi order limits and best practices.

6) **Execution APIs**
   - Use the official Kalshi create order endpoint `POST /portfolio/orders` for order placement. :contentReference[oaicite:7]{index=7}
   - Include robust signature/auth header signing per docs (API key + private key). :contentReference[oaicite:8]{index=8}
   - Implement cancel and list orders endpoints for lifecycle management.

7) **Integration of Polymarket ideas**
   - Add a **Spread Scanner** that ranks markets by how easy it is to place maker orders.
   - Add **Liquidity Ladder**:
     - Instead of only best price, quote 2–3 limit levels inside spread like a prediction bot might ladder bids/asks.
   - Add **“near-resolved filter” analog for Kalshi**:
     - Skip markets with time to close < threshold or prices near terminal values.

8) **Backtesting + simulation**
   - Implement a simple simulator that replays historical trades + snapshots and tests how your strategy would have behaved.

9) **Risk, logging, audit**
   - Record every decision, input signals, orders, fills to a ledger (SQLite or similar).
   - Enforce rate limits and exponential backoff on 429.

DELIVERABLES (CODE + CLI):
- Modules: market_scanner.py, data_ws.py, data_rest.py, order_manager.py, ev_gate.py
- CLI commands:
  - `kalshi-bot scan-markets --top <N>`
  - `kalshi-bot run-strategy --strategy maker_ladder --demo`
  - `kalshi-bot run-strategy --strategy maker_ladder --live --i-understand-risk`
  - `kalshi-bot backtest --market <ticker> --start ... --end ...`
- Tests for scanning, ev gating, order placement simulation.

REQUIREMENTS:
- Use only official Kalshi APIs (REST + WebSockets).
- Respect rate limits and backoff. :contentReference[oaicite:9]{index=9}
- Secure API key handling.
- Avoid near-terminated markets and extremely low liquidity.
- Log and audit everything.

Now implement all modules and CLI end-to-end with no TODOs.
