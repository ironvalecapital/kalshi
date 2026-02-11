from __future__ import annotations

import asyncio
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..config import BotSettings
from ..data_rest import KalshiDataClient
from ..decision_report import write_decision_report
from ..execution import ExecutionEngine, OrderRequest
from ..fee_model import fee_cents
from ..flow_features import FlowFeatures
from ..ledger import Ledger
from ..market_selector import pick_sports_candidates
from ..orderbook_live import LiveOrderbook, OrderbookState
from ..risk import RiskManager


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _fill_prob(depth: int, spread: int, trade_rate: float) -> float:
    x = 0.01 * depth - 0.15 * spread + 0.1 * trade_rate
    return _sigmoid(x)


def run_sports_strategy(
    settings: BotSettings,
    data_client: KalshiDataClient,
    ledger: Ledger,
    audit,
    risk: RiskManager,
    exec_engine: ExecutionEngine,
    cycles: int,
    sleep_s: int,
    live: bool,
) -> None:
    if live:
        audit.log("mode", "LIVE MODE", {})
    else:
        audit.log("mode", "DEMO MODE", {})

    loop_forever = cycles <= 0
    open_order_id: Optional[str] = None

    while True:
        if exec_engine._kill_switch():
            audit.log("kill", "kill switch enabled", {})
            break

        candidates = pick_sports_candidates(settings, data_client, top_n=settings.sports.top_n)
        if not candidates:
            audit.log("decision", "no sports candidates", {})
            time.sleep(sleep_s)
            if not loop_forever:
                cycles -= 1
                if cycles <= 0:
                    break
            continue

        pick = candidates[0]
        live_book = LiveOrderbook(settings, pick.ticker)
        flow = FlowFeatures()
        ob_state: Optional[OrderbookState] = None

        def on_update(state: OrderbookState):
            nonlocal ob_state
            ob_state = state

        def ws_thread():
            asyncio.run(live_book.run(on_update))

        t = threading.Thread(target=ws_thread, daemon=True)
        t.start()

        for _ in range(3):
            if ob_state:
                break
            time.sleep(1)

        if not ob_state:
            audit.log("decision", "no orderbook state", {"market": pick.ticker})
            time.sleep(sleep_s)
            continue

        yes_bid = ob_state.best_yes_bid()
        yes_ask = ob_state.best_yes_ask()
        no_bid = ob_state.best_no_bid()
        no_ask = ob_state.best_no_ask()
        spread = ob_state.spread_yes() or 0
        depth = ob_state.depth_topk(3)

        flow.update_mid((yes_bid + yes_ask) / 2 if yes_bid is not None and yes_ask is not None else 0)
        trades_resp = data_client.get_trades(ticker=pick.ticker, limit=50)
        trades = trades_resp.get("trades", [])
        now = datetime.now(timezone.utc)
        trades_5m = 0
        trades_60m = 0
        for tr in trades:
            ts = tr.get("ts") or tr.get("timestamp") or tr.get("time")
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
            if (now - dt).total_seconds() <= 300:
                trades_5m += 1
            if (now - dt).total_seconds() <= 3600:
                trades_60m += 1

        imbalance = flow.imbalance(depth_yes=depth // 2, depth_no=depth // 2)
        signed_vol = flow.signed_volume(60)
        delta_mid = flow.momentum(30)
        vol = flow.realized_var(60)

        a0, a1, a2, a3, a4, a5 = 0.0, 1.2, 0.01, 0.05, 0.3, 0.02
        p_next = _sigmoid(a0 + a1 * imbalance + a2 * signed_vol + a3 * delta_mid - a4 * spread - a5 * vol)

        implied_yes = (yes_ask / 100.0) if yes_ask is not None else 0.5
        edge_cents = (p_next - implied_yes) * 100.0
        fee = fee_cents(1, yes_ask or 0, maker=True)
        ev_after = edge_cents - fee

        fill_prob = _fill_prob(depth, spread, trades_5m / 5.0)
        action = "ABSTAIN"
        if ev_after >= settings.sports.min_ev_cents and spread <= settings.sports.max_spread_cents and fill_prob > 0.2:
            action = "BID_YES" if edge_cents >= 0 else "BID_NO"

        order_result: Dict[str, Any] = {}
        if action in ("BID_YES", "BID_NO"):
            price = (yes_bid or 0) + 1 if action == "BID_YES" else (no_bid or 0) + 1
            if yes_ask is not None and price >= yes_ask:
                price = yes_bid or 0
            if no_ask is not None and action == "BID_NO" and price >= no_ask:
                price = no_bid or 0
            size = min(settings.sports.max_order_size, max(settings.sports.base_size, int(ev_after / settings.sports.min_ev_cents)))
            ok, reason = risk.check_order(pick.ticker, size, price / 100.0)
            if ok:
                order = OrderRequest(
                    market_id=pick.ticker,
                    side="yes" if action == "BID_YES" else "no",
                    action="buy",
                    price_cents=price,
                    count=size,
                    client_order_id=f"sports-{int(time.time())}",
                )
                order_result = exec_engine.place_order(order)
            else:
                order_result = {"status": "rejected", "reason": reason}

        report = {
            "market_ticker": pick.ticker,
            "ts": datetime.now(timezone.utc).isoformat(),
            "best_yes_bid": yes_bid,
            "best_yes_ask": yes_ask,
            "best_no_bid": no_bid,
            "best_no_ask": no_ask,
            "spread": spread,
            "depth_top3": depth,
            "trade_rate_5m": trades_5m,
            "trade_rate_60m": trades_60m,
            "features": {
                "imbalance": imbalance,
                "signed_vol": signed_vol,
                "delta_mid_30s": delta_mid,
                "realized_var_1m": vol,
            },
            "model": {
                "p_next": p_next,
                "implied_yes": implied_yes,
                "edge_cents": edge_cents,
            },
            "fees": {"fee_cents": fee, "maker": True},
            "fill_prob": fill_prob,
            "action": action,
            "order_result": order_result,
        }

        write_decision_report(settings, report)
        ledger.record_decision(pick.ticker, "sports_orderflow", report, report.get("features", {}), ev_after, action, 1, {}, order_result)
        audit.log("decision", "sports orderflow", report)

        time.sleep(sleep_s)
        if not loop_forever:
            cycles -= 1
            if cycles <= 0:
                break
