from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dateutil import parser as date_parser
from rich.console import Console
from rich.table import Table

from .audit import AuditLogger
from .config import BotSettings, load_config
from .data_rest import KalshiDataClient, KalshiRestError
from .data_ws import KalshiWSClient
from .decision_report import write_decision_report
from .execution import ExecutionEngine, OrderRequest
from .ev import estimate_fee_cents
from .features import implied_probability, normalize_orderbook
from .ledger import Ledger
from .market_picker import pick_weather_candidates
from .models.bayes_prior import bayes_update, choose_prior
from .models.consistency import check_consistency
from .models.microstructure import adjust_for_event_time
from .rate_limit import RateLimiter, tier_to_limits
from .risk import RiskManager
from .strategies.weather_high_temp import run_weather_strategy

app = typer.Typer(add_completion=False)
console = Console()


def build_settings(config_path: Optional[str]) -> BotSettings:
    settings = load_config(config_path)
    limits = tier_to_limits(settings.rate_limit.tier)
    settings.rate_limit.read_per_sec = limits["read"]
    settings.rate_limit.write_per_sec = limits["write"]
    settings.rate_limit.burst = limits["burst"]
    return settings


def build_clients(settings: BotSettings):
    limiter = RateLimiter(settings.rate_limit.read_per_sec, settings.rate_limit.write_per_sec, settings.rate_limit.burst)
    data_client = KalshiDataClient(settings, limiter)
    return limiter, data_client


def ensure_demo_or_live(demo: bool, live: bool, confirm: bool) -> None:
    if live and demo:
        raise typer.BadParameter("Choose either --demo or --live")
    if not live:
        return
    if not confirm:
        raise typer.BadParameter("Live trading requires --i-understand-risk")



@app.command()
def doctor(config: Optional[str] = typer.Option(None, help="Path to YAML config")):
    settings = build_settings(config)
    limiter, data_client = build_clients(settings)
    console.print("Kalshi Decision Engine doctor")
    console.print(f"Env: {settings.data.env}")
    console.print(f"API base: {settings.data.api_base_demo if settings.data.env == 'demo' else settings.data.api_base_prod}")
    if not settings.api_key_id or not settings.private_key_path:
        console.print("API credentials missing (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH)")
    else:
        console.print("API credentials present")
    try:
        markets = data_client.list_markets(limit=1)
        console.print(f"Connectivity: OK (markets={len(markets.get('markets', []))})")
    except Exception as exc:
        console.print(f"Connectivity: FAILED ({exc})")
    if settings.api_key_id and settings.private_key_path:
        try:
            balance = data_client.get_balance()
            console.print(f"SDK balance check: {'OK' if balance is not None else 'Unavailable'}")
        except Exception as exc:
            console.print(f"SDK balance check: FAILED ({exc})")
    console.print("Rate limit tier:", settings.rate_limit.tier)


@app.command()
def markets(
    query: str = typer.Option("", help="Search keyword"),
    top: int = typer.Option(20, help="Max results"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    data = data_client.list_markets(query=query or None, limit=top)
    table = Table(title="Markets")
    table.add_column("Ticker")
    table.add_column("Title")
    table.add_column("Status")
    for m in data.get("markets", []):
        table.add_row(m.get("ticker", ""), m.get("title", ""), m.get("status", ""))
    console.print(table)


@app.command()
def watch(
    market: str = typer.Option(..., help="Market ticker"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    ws = KalshiWSClient(settings)

    async def _watch():
        async for msg in ws.stream_ticker([market]):
            console.print(msg)

    asyncio.run(_watch())


@app.command()
def pick_weather(
    top: int = typer.Option(20, help="Top N"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    candidates = pick_weather_candidates(settings, data_client, top_n=top)
    if not candidates:
        console.print("No suitable market found.")
        raise typer.Exit(0)
    table = Table(title="Weather Market Pick")
    table.add_column("Ticker")
    table.add_column("Close")
    table.add_column("Spread")
    table.add_column("Trades(1h)")
    table.add_column("Volume")
    for c in candidates:
        table.add_row(
            c.ticker,
            c.close_time.isoformat(),
            str(c.spread_yes or ""),
            str(c.trades_1h),
            f"{c.volume:.0f}",
        )
    console.print(table)


@app.command()
def run_weather(
    demo: bool = typer.Option(True, help="Use demo environment"),
    live: bool = typer.Option(False, help="Use live environment"),
    i_understand_risk: bool = typer.Option(False, help="Confirm live trading"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    cycles: int = typer.Option(1, help="Number of cycles"),
    sleep: int = typer.Option(10, help="Seconds between cycles"),
):
    if live:
        demo = False
    ensure_demo_or_live(demo, live, i_understand_risk)
    settings = build_settings(config)
    settings.data.env = "prod" if live else "demo"
    console.print("DEMO MODE" if not live else "LIVE MODE")
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    audit = AuditLogger(ledger, settings.log_path)
    risk = RiskManager(settings.risk)
    exec_engine = ExecutionEngine(data_client, ledger, risk, settings.execution)
    overrides = settings.weather.market_overrides or {}
    run_weather_strategy(settings, data_client, ledger, audit, risk, exec_engine, overrides, cycles, sleep, live)


@app.command()
def backtest(
    market: str = typer.Option(..., help="Market ticker"),
    start: str = typer.Option(..., help="Start ISO timestamp"),
    end: str = typer.Option(..., help="End ISO timestamp"),
    strategy: str = typer.Option("bayes", help="bayes|consistency|micro"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    ledger = Ledger(settings.db_path)
    rows = ledger.get_ticks(market, start, end)
    if not rows:
        console.print("No ticks found in ledger for that range.")
        raise typer.Exit(0)
    pnl = 0.0
    trades = 0
    for row in rows:
        mid = row["mid"]
        if mid is None:
            continue
        implied = implied_probability(mid)
        if implied is None:
            continue
        prior = choose_prior()
        result = bayes_update(prior, implied)
        edge = result.edge_cents
        if edge > settings.execution.require_edge_cents:
            pnl += edge / 100.0
            trades += 1
        elif edge < -settings.execution.require_edge_cents:
            pnl += (-edge) / 100.0
            trades += 1
    console.print(f"Backtest trades: {trades}")
    console.print(f"Backtest edge PnL (approx): {pnl:.2f} USD")


@app.command()
def run(
    strategy: str = typer.Option("bayes", help="bayes|consistency|micro"),
    market: str = typer.Option(..., help="Market ticker"),
    demo: bool = typer.Option(True, help="Use demo environment"),
    live: bool = typer.Option(False, help="Use live environment"),
    i_understand_risk: bool = typer.Option(False, help="Confirm live trading"),
    dry_run: bool = typer.Option(False, help="Compute decisions without placing orders"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    once: bool = typer.Option(False, help="Run single decision cycle"),
    interval: int = typer.Option(10, help="Seconds between cycles"),
):
    ensure_demo_or_live(demo, live, i_understand_risk)
    settings = build_settings(config)
    settings.data.env = "prod" if live else "demo"

    limiter, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    audit = AuditLogger(ledger, settings.log_path)
    risk = RiskManager(settings.risk)
    exec_engine = ExecutionEngine(data_client, ledger, risk, settings.execution)

    while True:
        paused, reason = risk.should_pause()
        if paused:
            audit.log("pause", "trading paused", {"reason": reason})
            time.sleep(interval)
            if once:
                break
            continue
        try:
            if settings.api_key_id and settings.private_key_path:
                orders = data_client.get_orders(status="open")
                open_orders = len(orders.get("orders", []))
                risk.note_open_orders(open_orders)
                positions = data_client.get_positions()
                for pos in positions.get("positions", []):
                    ticker = pos.get("market_ticker") or pos.get("ticker")
                    notional = float(pos.get("notional", 0) or 0)
                    if ticker:
                        risk.note_position(ticker, notional)
            market_info = data_client.get_market(market)
            event_id = market_info.get("event_ticker") or market_info.get("event_id")
            close_time_raw = market_info.get("close_time") or market_info.get("close_ts")
            close_time = date_parser.parse(close_time_raw) if close_time_raw else None
            orderbook = data_client.get_orderbook(market)
            snap = normalize_orderbook(market, event_id, orderbook)
            ledger.record_tick(
                {
                    "market_id": market,
                    "event_id": event_id,
                    "timestamp": snap.timestamp.isoformat(),
                    "bid": snap.bid,
                    "ask": snap.ask,
                    "mid": snap.mid,
                    "spread": snap.spread,
                    "volume": snap.volume,
                    "open_interest": snap.open_interest,
                    "source": "rest",
                }
            )
            implied = implied_probability(snap.mid)
            if implied is None:
                audit.log("decision", "no implied prob", {"market": market})
                if once:
                    break
                time.sleep(interval)
                continue

            prior = choose_prior()
            bayes = bayes_update(prior, implied)
            edge_raw = bayes.edge_cents

            signals: Dict[str, Any] = {
                "implied": implied,
                "prior": prior,
                "posterior": bayes.posterior,
                "fair_value_cents": bayes.fair_value_cents,
                "edge_cents": edge_raw,
            }

            required_edge = settings.execution.require_edge_cents
            edge = edge_raw
            size_multiplier = 1.0
            if strategy == "micro":
                micro = adjust_for_event_time(close_time, required_edge)
                required_edge = micro.required_edge_cents
                size_multiplier = micro.size_multiplier
                signals["microstructure"] = micro.__dict__

            if strategy == "consistency":
                related = data_client.list_markets(query=event_id, limit=5)
                implieds: List[float] = []
                for m in related.get("markets", []):
                    try:
                        ob = data_client.get_orderbook(m.get("ticker"))
                        ms = normalize_orderbook(m.get("ticker"), event_id, ob)
                        ip = implied_probability(ms.mid)
                        if ip is not None:
                            implieds.append(ip)
                    except Exception:
                        continue
                consistency = check_consistency(implieds)
                signals["consistency"] = consistency.__dict__
                if not consistency.consistent:
                    ledger.record_decision(market, strategy, {"mid": snap.mid}, signals, edge, "abstain", 0, {"consistent": False}, {})
                    audit.log("decision", "abstain due to inconsistency", {"market": market})
                    if once:
                        break
                    time.sleep(interval)
                    continue

            if snap.spread is not None and settings.risk.pause_on_spread_wide:
                if snap.spread > settings.risk.spread_wide_threshold_cents:
                    audit.log("decision", "spread too wide", {"spread": snap.spread})
                    if once:
                        break
                    time.sleep(interval)
                    continue

            action = "hold"
            side = "yes"
            size = max(1, int(1 * size_multiplier))
            size = min(size, settings.risk.max_order_size_contracts)
            price_cents = int(snap.bid or 0)
            if edge >= required_edge:
                action = "buy"
                side = "yes"
                price_cents = int(snap.bid or 0)
            elif edge <= -required_edge:
                action = "buy"
                side = "no"
                price_cents = int(snap.no_bid or 0)
            else:
                action = "hold"

            order_result: Dict[str, Any] = {}
            fee_estimate_cents = estimate_fee_cents(
                count=size,
                price_cents=price_cents,
                maker=settings.execution.prefer_maker,
                maker_rate=settings.execution.maker_fee_rate,
                taker_rate=settings.execution.taker_fee_rate,
            )
            edge_after_fees = edge_raw - fee_estimate_cents
            signals["fee_estimate_cents"] = fee_estimate_cents
            signals["edge_after_fees_cents"] = edge_after_fees

            if action != "hold" and edge_after_fees < required_edge:
                action = "hold"
                order_result = {"status": "rejected", "reason": "edge_after_fees_below_threshold"}

            risk_checks = {"required_edge": required_edge, "edge": edge_raw, "edge_after_fees": edge_after_fees, "spread": snap.spread}
            if action != "hold":
                if price_cents <= 0:
                    action = "hold"
                    order_result = {"status": "rejected", "reason": "invalid_price"}
                elif dry_run:
                    order_result = {"status": "dry_run", "reason": "dry_run_enabled"}
                else:
                    order = OrderRequest(
                        market_id=market,
                        side=side,
                        action=action,
                        price_cents=price_cents,
                        count=size,
                        client_order_id=f"kalshi-bot-{int(time.time())}",
                    )
                    order_result = exec_engine.place_order(order)
            ledger.record_decision(market, strategy, {"mid": snap.mid}, signals, edge_after_fees, action, size, risk_checks, order_result)
            report = {
                "inputs": {"market": market, "mid": snap.mid, "spread": snap.spread},
                "signals": signals,
                "expected_edge": edge_after_fees,
                "chosen_action": action,
                "risk_checks": risk_checks,
                "order_result": order_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            report_path = write_decision_report(settings, report)
            audit.log("decision", "decision cycle complete", {"report": str(report_path)})
        except KalshiRestError as exc:
            audit.log("error", "api error", {"error": str(exc), "status_code": exc.status_code})
            if exc.status_code in (429, 403):
                risk.note_api_error(exc.status_code, exc.retry_after)
        except Exception as exc:
            audit.log("error", "unexpected error", {"error": str(exc)})
        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    app()
