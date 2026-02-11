from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from pathlib import Path
from dateutil import parser as date_parser
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..config import BotSettings
from ..data_rest import KalshiDataClient
from ..decision_report import write_decision_report
from ..execution import ExecutionEngine, OrderRequest
from ..ev import kelly_contracts, kelly_fraction_no, kelly_fraction_yes
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


def _implied_mid_yes(yes_bid: Optional[int], yes_ask: Optional[int]) -> Optional[float]:
    if yes_bid is None or yes_ask is None:
        return None
    return (yes_bid + yes_ask) / 2.0 / 100.0


def _microprice_yes(yes_bid: Optional[int], yes_ask: Optional[int], depth_yes: int, depth_no: int) -> Optional[float]:
    if yes_bid is None or yes_ask is None:
        return None
    denom = depth_yes + depth_no
    if denom <= 0:
        return None
    micro = (yes_bid * depth_yes + yes_ask * depth_no) / denom
    return micro / 100.0


def _vwap_yes(trades: list[dict], window_sec: int = 300) -> Optional[float]:
    if not trades:
        return None
    now = datetime.now(timezone.utc)
    notional = 0.0
    volume = 0.0
    for tr in trades:
        ts = tr.get("ts") or tr.get("timestamp") or tr.get("time") or tr.get("created_time")
        if ts is None:
            continue
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        if (now - dt).total_seconds() > window_sec:
            continue
        price = tr.get("yes_price") or tr.get("price")
        count = tr.get("count") or 0
        if price is None or count <= 0:
            continue
        notional += float(price) * float(count)
        volume += float(count)
    if volume <= 0:
        return None
    return (notional / volume) / 100.0


def _auto_pick_from_summary(settings: BotSettings, data_client: KalshiDataClient) -> Optional[str]:
    resp = data_client.list_markets(status="open", limit=1000)
    markets = resp.get("markets", [])
    markets = sorted(
        markets,
        key=lambda m: float(m.get("volume_24h", 0) or m.get("volume", 0) or 0),
        reverse=True,
    )[: settings.sports.auto_pick_top_n]
    for m in markets:
        if not settings.sports.auto_pick_use_summary:
            continue
        if m.get("yes_bid") is not None or m.get("no_bid") is not None:
            return m.get("ticker")
    return None


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
    market_override: Optional[str] = None,
) -> None:
    if live:
        audit.log("mode", "LIVE MODE", {})
    else:
        audit.log("mode", "DEMO MODE", {})

    loop_forever = cycles <= 0
    open_order_id: Optional[str] = None
    last_report_ts = time.time()
    cycle_stats = {"decisions": 0, "orders": 0, "abstains": 0}
    last_edge_by_market: dict[str, float] = {}

    while True:
        if exec_engine._kill_switch():
            audit.log("kill", "kill switch enabled", {})
            break

        pick = None
        if market_override:
            pick = type("Pick", (), {"ticker": market_override, "event_ticker": ""})()
        else:
            auto_ticker = _auto_pick_from_summary(settings, data_client)
            if auto_ticker:
                pick = type("Pick", (), {"ticker": auto_ticker, "event_ticker": ""})()
            else:
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
            # Fall back to market summary quotes if WS/orderbook unavailable.
            market_info = data_client.get_market(pick.ticker)
            yes_bid = market_info.get("yes_bid")
            yes_ask = market_info.get("yes_ask")
            no_bid = market_info.get("no_bid")
            no_ask = market_info.get("no_ask")
            if yes_bid is None and yes_ask is None and no_bid is None and no_ask is None:
                time.sleep(sleep_s)
                continue
            depth_yes = 0
            depth_no = 0
        else:
            yes_bid = ob_state.best_yes_bid()
            yes_ask = ob_state.best_yes_ask()
            no_bid = ob_state.best_no_bid()
            no_ask = ob_state.best_no_ask()
            depth_yes = ob_state.depth_yes_topk(3)
            depth_no = ob_state.depth_no_topk(3)

        if ob_state:
            spread = ob_state.spread_yes() or 0
        else:
            spread = (yes_ask - yes_bid) if yes_ask is not None and yes_bid is not None else 0
        depth = depth_yes + depth_no

        if yes_bid is None and no_bid is None:
            audit.log("decision", "empty orderbook", {"market": pick.ticker})
            time.sleep(sleep_s)
            if not loop_forever:
                cycles -= 1
                if cycles <= 0:
                    break
            continue

        if yes_bid is not None and yes_ask is not None:
            flow.update_mid((yes_bid + yes_ask) / 2)
        flow.update_book(yes_bid, yes_ask, spread)
        trades_resp = data_client.get_trades(ticker=pick.ticker, limit=50)
        trades = trades_resp.get("trades", [])
        now = datetime.now(timezone.utc)
        trades_5m = 0
        trades_60m = 0
        for tr in trades:
            ts = tr.get("ts") or tr.get("timestamp") or tr.get("time") or tr.get("created_time")
            if ts is None:
                continue
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
            side = tr.get("taker_side") or tr.get("side")
            price = tr.get("yes_price") or tr.get("price") or 0
            count = tr.get("count") or 0
            if price and count:
                flow.update_trade(price=price, size=count, side=side, ts=dt)
            if (now - dt).total_seconds() <= 300:
                trades_5m += 1
            if (now - dt).total_seconds() <= 3600:
                trades_60m += 1

        imbalance = flow.imbalance(depth_yes=depth_yes, depth_no=depth_no)
        signed_vol = flow.signed_volume(60)
        delta_mid = flow.momentum(30)
        bid_mom = flow.bid_momentum(30)
        ask_mom = flow.ask_momentum(30)
        spread_trend = flow.spread_trend(30)
        vol = flow.realized_var(60)

        a0, a1, a2, a3, a4, a5 = 0.0, 1.2, 0.01, 0.05, 0.3, 0.02
        p_next = _sigmoid(a0 + a1 * imbalance + a2 * signed_vol + a3 * delta_mid - a4 * spread - a5 * vol)

        implied_mid = _implied_mid_yes(yes_bid, yes_ask)
        micro = _microprice_yes(yes_bid, yes_ask, depth_yes, depth_no)
        vwap = _vwap_yes(trades, window_sec=300)

        # Base implied probability from mid, then adjust by orderflow signals.
        implied_yes = implied_mid if implied_mid is not None else (yes_ask / 100.0 if yes_ask is not None else 0.5)
        drift = 0.0
        if micro is not None and implied_mid is not None:
            drift += (micro - implied_mid)
        if vwap is not None and implied_mid is not None:
            drift += (vwap - implied_mid)
        # Convert feature mix into a small probability delta.
        delta_p = 0.05 * imbalance + 0.002 * delta_mid + 0.004 * bid_mom - 0.004 * ask_mom - 0.003 * spread_trend
        delta_p += 0.01 * (drift * 100.0)
        p_next = max(0.01, min(0.99, implied_yes + delta_p))

        edge_cents = (p_next - (yes_ask / 100.0 if yes_ask is not None else implied_yes)) * 100.0
        fee = fee_cents(1, yes_ask or 0, maker=True)
        ev_after = edge_cents - fee

        fill_prob = _fill_prob(depth, spread, trades_5m / 5.0)
        action = "ABSTAIN"
        min_ev = settings.sports.min_ev_cents
        # Category gating: higher EV threshold for sports/entertainment/media-like categories.
        if pick.event_ticker:
            key = str(pick.event_ticker).upper()
            if any(tag in key for tag in ["SPORT", "NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "ENT", "MEDIA"]):
                min_ev *= settings.sports.category_ev_multiplier
        if ev_after >= min_ev and spread <= settings.sports.max_spread_cents and fill_prob > 0.1:
            action = "BID_YES" if edge_cents >= 0 else "BID_NO"

        # Longshot bias filters: avoid buying YES at extreme low prices; prefer NO at tails.
        if action == "BID_YES" and yes_ask is not None and yes_ask <= settings.sports.yes_longshot_max_cents:
            action = "ABSTAIN"
        if action == "BID_NO" and yes_ask is not None and yes_ask < settings.sports.no_tail_min_cents:
            action = "ABSTAIN"

        order_result: Dict[str, Any] = {}
        if action in ("BID_YES", "BID_NO"):
            price = (yes_bid or 0) + 1 if action == "BID_YES" else (no_bid or 0) + 1
            if settings.sports.maker_only:
                if yes_ask is not None and price >= yes_ask:
                    price = yes_bid or 0
                if no_ask is not None and action == "BID_NO" and price >= no_ask:
                    price = no_bid or 0
            fee_per = fee_cents(1, price, maker=True)
            if action == "BID_YES":
                kfrac = kelly_fraction_yes(p_next, price, fee_per, 0.0)
            else:
                kfrac = kelly_fraction_no(p_next, price, fee_per, 0.0)
            size = kelly_contracts(
                bankroll_usd=settings.execution.bankroll_usd,
                price_cents=price,
                kelly_fraction=kfrac,
                fractional=settings.execution.kelly_fraction,
                fill_prob=fill_prob,
                use_fill_prob=settings.execution.kelly_use_fill_prob,
                max_contracts=settings.sports.max_order_size,
            )
            if size <= 0:
                order_result = {"status": "rejected", "reason": "kelly_size_zero"}
            else:
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
            "depth_yes_top3": depth_yes,
            "depth_no_top3": depth_no,
            "trade_rate_5m": trades_5m,
            "trade_rate_60m": trades_60m,
            "features": {
                "imbalance": imbalance,
                "signed_vol": signed_vol,
                "delta_mid_30s": delta_mid,
                "bid_momentum_30s": bid_mom,
                "ask_momentum_30s": ask_mom,
                "spread_trend_30s": spread_trend,
                "microprice": micro,
                "vwap_5m": vwap,
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
        last_edge_by_market[pick.ticker] = edge_cents
        cycle_stats["decisions"] += 1
        if action == "ABSTAIN":
            cycle_stats["abstains"] += 1
        if order_result.get("status") == "submitted":
            cycle_stats["orders"] += 1

        now = time.time()
        if now - last_report_ts >= settings.sports.daily_report_interval_sec:
            report_path = Path(settings.decision_report_dir) / f"daily_report_{time.strftime('%Y%m%d_%H%M')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "market": pick.ticker,
                "decisions": cycle_stats["decisions"],
                "orders": cycle_stats["orders"],
                "abstains": cycle_stats["abstains"],
            }
            report_path.write_text(json.dumps(payload, indent=2))
            last_report_ts = now
            cycle_stats = {"decisions": 0, "orders": 0, "abstains": 0}

        # Stale order cancellation
        try:
            orders = data_client.get_orders(status="open")
            for o in orders.get("orders", []):
                ticker = o.get("ticker") or o.get("market_ticker")
                if not ticker:
                    continue
                edge = last_edge_by_market.get(ticker, 0.0)
                created = o.get("created_time") or o.get("created_ts") or o.get("time")
                if created is None:
                    continue
                if isinstance(created, (int, float)):
                    age = time.time() - float(created)
                else:
                    dt = date_parser.parse(str(created).replace("Z", "+00:00"))
                    age = time.time() - dt.timestamp()
                if age >= settings.sports.stale_order_max_age_sec and edge < settings.sports.stale_order_cancel_edge_cents:
                    exec_engine.cancel_order(o.get("order_id"))
                    audit.log("cancel", "stale order cancelled", {"order_id": o.get("order_id"), "market": ticker, "age": age, "edge": edge})
        except Exception as exc:
            audit.log("cancel", "stale order check failed", {"error": str(exc)})

        time.sleep(sleep_s)
        if not loop_forever:
            cycles -= 1
            if cycles <= 0:
                break
