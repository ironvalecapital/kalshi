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
from .market_scanner import scan_markets as scan_markets_fn
from .models.bayes_prior import bayes_update, choose_prior
from .models.consistency import check_consistency
from .models.microstructure import adjust_for_event_time
from .market_selector import pick_sports_candidates
from .orderbook_live import LiveOrderbook
from .flow_features import FlowFeatures
from .strategies.sports_orderflow import run_sports_strategy
from .ingest.orderflow import sync_trades
from .tools.breakeven import breakeven_yes, breakeven_no
from .rate_limit import RateLimiter, tier_to_limits
from .risk import RiskManager
from .strategies.weather_high_temp import run_weather_strategy
from .automate.learner import run_learn
from .spread_scanner import scan_spreads as scan_spreads_fn
from .edge_scorer import score_no, score_yes, implied_probability as implied_prob_score
from .execution_manager import maker_ladder_cycle
from .order_lifecycle import OrderLifecycle
from .edge_scorer import score_no, score_yes, implied_probability
from .execution_manager import maker_ladder_cycle
from .order_lifecycle import OrderLifecycle
from .watchlist import build_watchlist
from .watchlist_server import serve_watchlist

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
        markets = data_client.list_markets(limit=50, status="open")
        open_count = len(markets.get("markets", []))
        if open_count == 0:
            unopened = data_client.list_markets(limit=50, status="unopened")
            unopened_count = len(unopened.get("markets", []))
            if unopened_count > 0:
                console.print(
                    f"Connectivity: DEGRADED (open_markets_sample=0, unopened_sample={unopened_count})"
                )
            else:
                console.print(
                    "Connectivity: DEGRADED (market samples are empty; likely rate-limited or temporary API throttling)"
                )
        else:
            console.print(f"Connectivity: OK (open_markets_sample={open_count})")
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
    market: Optional[str] = typer.Option(None, help="Market ticker"),
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
def pick_sports(
    top: int = typer.Option(20, help="Top N"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    # Keep interactive picker fast under rate limits.
    settings.sports.orderbook_probe_limit = min(settings.sports.orderbook_probe_limit, max(20, top * 2))
    settings.sports.selector_workers = 1
    _, data_client = build_clients(settings)
    candidates = pick_sports_candidates(settings, data_client, top_n=top)
    if not candidates:
        console.print("No suitable sports markets found.")
        raise typer.Exit(0)
    table = Table(title="Sports Market Pick")
    table.add_column("Ticker")
    table.add_column("Spread")
    table.add_column("Trades(60m)")
    table.add_column("DepthTop3")
    for c in candidates:
        table.add_row(c.ticker, str(c.spread_yes), str(c.trades_60m), str(c.depth_top3))
    console.print(table)


@app.command()
def watchlist(
    top: int = typer.Option(20, help="Top N"),
    include_weather: bool = typer.Option(True, help="Include weather lane"),
    include_sports: bool = typer.Option(True, help="Include sports lane"),
    loop: bool = typer.Option(False, help="Continuously refresh"),
    interval: int = typer.Option(15, help="Seconds between refresh"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    while True:
        items = build_watchlist(
            settings,
            data_client,
            top=top,
            include_weather=include_weather,
            include_sports=include_sports,
        )
        table = Table(title="Watchlist")
        table.add_column("Lane")
        table.add_column("Ticker")
        table.add_column("Title")
        table.add_column("Status")
        table.add_column("Spread")
        table.add_column("Trades(1h)")
        table.add_column("Trades(60m)")
        table.add_column("DepthTop3")
        table.add_column("Close")
        for i in items:
            table.add_row(
                i.lane,
                i.ticker,
                i.title or "",
                i.status or "",
                "" if i.spread_cents is None else str(i.spread_cents),
                "" if i.trades_1h is None else str(i.trades_1h),
                "" if i.trades_60m is None else str(i.trades_60m),
                "" if i.depth_top3 is None else str(i.depth_top3),
                i.close_time or "",
            )
        console.clear()
        console.print(table)
        if not loop:
            break
        time.sleep(interval)


@app.command()
def watchlist_server(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Port"),
    top: int = typer.Option(20, help="Top N"),
    include_weather: bool = typer.Option(True, help="Include weather lane"),
    include_sports: bool = typer.Option(True, help="Include sports lane"),
    refresh: int = typer.Option(30, help="Seconds between refresh"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    console.print(f"Serving watchlist on http://{host}:{port}")
    serve_watchlist(
        settings,
        data_client,
        host=host,
        port=port,
        top=top,
        include_weather=include_weather,
        include_sports=include_sports,
        refresh_sec=refresh,
    )


@app.command()
def watch_flow(
    market: str = typer.Option(..., help="Market ticker"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    flow = FlowFeatures()
    book = LiveOrderbook(settings, market)

    async def _run():
        async def on_update(state):
            best_yes = state.best_yes_bid()
            best_yes_ask = state.best_yes_ask()
            if best_yes is not None and best_yes_ask is not None:
                mid = (best_yes + best_yes_ask) / 2
                flow.update_mid(mid)
            depth_yes = state.depth_yes_topk(3)
            depth_no = state.depth_no_topk(3)
            console.print(
                {
                    "market": market,
                    "best_yes": best_yes,
                    "best_yes_ask": best_yes_ask,
                    "spread": state.spread_yes(),
                    "imbalance": flow.imbalance(depth_yes, depth_no),
                    "momentum_30s": flow.momentum(30),
                }
            )

        await book.run(on_update)

    asyncio.run(_run())


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
    if live and os.getenv("KALSHI_ARM_LIVE", "0") not in ("1", "true", "TRUE"):
        raise typer.BadParameter("Live trading requires KALSHI_ARM_LIVE=1")
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
def run_sports(
    demo: bool = typer.Option(True, help="Use demo environment"),
    live: bool = typer.Option(False, help="Use live environment"),
    i_understand_risk: bool = typer.Option(False, help="Confirm live trading"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    cycles: int = typer.Option(1, help="Number of cycles"),
    sleep: int = typer.Option(10, help="Seconds between cycles"),
    market: Optional[str] = typer.Option(None, help="Override market ticker"),
):
    if live:
        demo = False
    ensure_demo_or_live(demo, live, i_understand_risk)
    if live and os.getenv("KALSHI_ARM_LIVE", "0") not in ("1", "true", "TRUE"):
        raise typer.BadParameter("Live trading requires KALSHI_ARM_LIVE=1")
    settings = build_settings(config)
    settings.data.env = "prod" if live else "demo"
    console.print("DEMO MODE" if not live else "LIVE MODE")
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    audit = AuditLogger(ledger, settings.log_path)
    risk = RiskManager(settings.risk)
    exec_engine = ExecutionEngine(data_client, ledger, risk, settings.execution)
    run_sports_strategy(settings, data_client, ledger, audit, risk, exec_engine, cycles, sleep, live, market_override=market)


@app.command()
def ingest_trades(
    ticker: str = typer.Option(..., help="Market ticker to ingest trades for"),
    lookback: int = typer.Option(3600, help="Lookback seconds to fetch trades"),
    limit: int = typer.Option(200, help="Trades page size"),
    max_pages: int = typer.Option(20, help="Max pages to fetch"),
    loop: bool = typer.Option(False, help="Continuously ingest"),
    sleep: int = typer.Option(30, help="Seconds between ingest cycles"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    """Ingest recent trades (orderflow tape) into the SQLite ledger."""
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    while True:
        stored, total = sync_trades(
            data_client,
            ledger,
            ticker=ticker,
            lookback_sec=lookback,
            limit=limit,
            max_pages=max_pages,
        )
        console.print(f"Ingested {stored} trades (fetched {total}) for {ticker}")
        if not loop:
            break
        time.sleep(sleep)


@app.command()
def breakeven(
    price: Optional[int] = typer.Option(None, help="Entry price in cents"),
    side: str = typer.Option("yes", help="yes|no"),
    count: int = typer.Option(1, help="Contract count"),
    maker: bool = typer.Option(False, help="Assume maker fee"),
    taker: bool = typer.Option(False, help="Assume taker fee"),
    slip: float = typer.Option(0.0, help="Assumed slippage in cents"),
    assume_spread: Optional[int] = typer.Option(None, help="Assumed spread in cents (for taker)"),
    ticker: Optional[str] = typer.Option(None, help="Market ticker"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    if maker and taker:
        raise typer.BadParameter("Choose either --maker or --taker")
    maker_mode = maker or not taker
    if ticker:
        _, data_client = build_clients(settings)
        ob = data_client.get_orderbook(ticker)
        yes_bids = ob.get("yes", [])
        no_bids = ob.get("no", [])
        best_yes_bid = yes_bids[0][0] if yes_bids else None
        best_no_bid = no_bids[0][0] if no_bids else None
        yes_ask = 100 - best_no_bid if best_no_bid is not None else None
        no_ask = 100 - best_yes_bid if best_yes_bid is not None else None
        spread_yes = (yes_ask - best_yes_bid) if yes_ask is not None and best_yes_bid is not None else None
        rows = []
        if yes_ask is not None and best_yes_bid is not None:
            maker_price = min(yes_ask - 1, best_yes_bid + 1)
            rows.append(("maker", "YES", maker_price, breakeven_yes(maker_price, count, True, slip)))
            rows.append(("taker", "YES", yes_ask, breakeven_yes(yes_ask, count, False, (assume_spread or (spread_yes or 0)) / 2)))
        if no_ask is not None and best_no_bid is not None:
            maker_price = min(no_ask - 1, best_no_bid + 1)
            rows.append(("maker", "NO", maker_price, breakeven_no(maker_price, count, True, slip)))
            rows.append(("taker", "NO", no_ask, breakeven_no(no_ask, count, False, (assume_spread or (spread_yes or 0)) / 2)))
        table = Table(title=f"Breakeven {ticker}")
        table.add_column("Mode")
        table.add_column("Side")
        table.add_column("Entry")
        table.add_column("Fee(total)")
        table.add_column("Fee/contract")
        table.add_column("Slip")
        table.add_column("p*")
        table.add_column("p_mkt")
        table.add_column("Δp")
        table.add_column("Edge(c)")
        for mode, side_label, entry, r in rows:
            p_star = r.get("p_break_even") or r.get("p_break_even_no")
            p_mkt = r.get("p_market") or r.get("p_market_no")
            table.add_row(
                mode,
                side_label,
                str(entry),
                f"{r['fee_total_cents']:.0f}",
                f"{r['fee_per_contract_cents']:.2f}",
                f"{r['slip_cents']:.2f}",
                f"{p_star:.4f}",
                f"{p_mkt:.4f}",
                f"{r['delta_p']:.4f}",
                f"{r['edge_cents']:.2f}",
            )
        console.print(table)
        return

    if price is None:
        raise typer.BadParameter("Provide --price or --ticker")
    side = side.lower()
    if side not in ("yes", "no"):
        raise typer.BadParameter("--side must be yes or no")
    if taker and assume_spread is not None:
        slip = assume_spread / 2
    if side == "yes":
        r = breakeven_yes(price, count, maker_mode, slip)
    else:
        r = breakeven_no(price, count, maker_mode, slip)
    table = Table(title="Breakeven")
    table.add_column("Mode")
    table.add_column("Side")
    table.add_column("Entry")
    table.add_column("Fee(total)")
    table.add_column("Fee/contract")
    table.add_column("Slip")
    table.add_column("p*")
    table.add_column("p_mkt")
    table.add_column("Δp")
    table.add_column("Edge(c)")
    p_star = r.get("p_break_even") or r.get("p_break_even_no")
    p_mkt = r.get("p_market") or r.get("p_market_no")
    table.add_row(
        "maker" if maker_mode else "taker",
        side.upper(),
        str(price),
        f"{r['fee_total_cents']:.0f}",
        f"{r['fee_per_contract_cents']:.2f}",
        f"{r['slip_cents']:.2f}",
        f"{p_star:.4f}",
        f"{p_mkt:.4f}",
        f"{r['delta_p']:.4f}",
        f"{r['edge_cents']:.2f}",
    )
    console.print(table)

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
    lane: Optional[str] = typer.Option(None, help="Lane: weather|sports"),
    strategy: str = typer.Option("bayes", help="bayes|consistency|micro"),
    market: Optional[str] = typer.Option(None, help="Market ticker"),
    demo: bool = typer.Option(True, help="Use demo environment"),
    live: bool = typer.Option(False, help="Use live environment"),
    i_understand_risk: bool = typer.Option(False, help="Confirm live trading"),
    dry_run: bool = typer.Option(False, help="Compute decisions without placing orders"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    once: bool = typer.Option(False, help="Run single decision cycle"),
    interval: int = typer.Option(10, help="Seconds between cycles"),
):
    if lane:
        if live:
            demo = False
        ensure_demo_or_live(demo, live, i_understand_risk)
        if live and os.getenv("KALSHI_ARM_LIVE", "0") not in ("1", "true", "TRUE"):
            raise typer.BadParameter("Live trading requires KALSHI_ARM_LIVE=1")
        if lane == "weather":
            run_weather(demo=demo, live=live, i_understand_risk=i_understand_risk, config=config, cycles=1, sleep=interval)
            return
        if lane == "sports":
            run_sports(demo=demo, live=live, i_understand_risk=i_understand_risk, config=config, cycles=1, sleep=interval, market=None)
            return
        raise typer.BadParameter("lane must be weather or sports")

    if market is None:
        raise typer.BadParameter("Market ticker required unless using --lane")
    ensure_demo_or_live(demo, live, i_understand_risk)
    if live and os.getenv("KALSHI_ARM_LIVE", "0") not in ("1", "true", "TRUE"):
        raise typer.BadParameter("Live trading requires KALSHI_ARM_LIVE=1")
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


@app.command()
def learn(
    lane: str = typer.Option("weather", help="Learning lane (weather only for now)"),
    demo: bool = typer.Option(True, help="Always demo"),
    loop: bool = typer.Option(True, help="Run continuously"),
    interval: int = typer.Option(300, help="Seconds between learn cycles"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    if not demo:
        raise typer.BadParameter("Learning runs in demo mode only.")
    settings = build_settings(config)
    settings.data.env = "demo"
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    audit = AuditLogger(ledger, settings.log_path)
    run_learn(settings, data_client, ledger, audit, lane=lane, loop=loop, interval_sec=interval)


@app.command()
def report(
    latest: bool = typer.Option(True, help="Show latest learning summary"),
):
    from .living_files import write_operating_rules

    write_operating_rules()
    path = Path("living_files_kalshi/MEMORY_PACK.json")
    if not path.exists():
        console.print("No MEMORY_PACK.json yet.")
        raise typer.Exit(0)
    console.print(path.read_text())


@app.command()
def scan_spreads_cmd(
    top: int = typer.Option(20, help="Top N"),
    min_spread: int = typer.Option(2, help="Min spread (cents)"),
    max_spread: int = typer.Option(30, help="Max spread (cents)"),
    status: str = typer.Option("open", help="Market status"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    items = scan_spreads_fn(settings, data_client, top=top, min_spread=min_spread, max_spread=max_spread, status=status)
    table = Table(title="Spread Scanner")
    table.add_column("Ticker")
    table.add_column("Spread")
    table.add_column("YesBid")
    table.add_column("YesAsk")
    table.add_column("DepthTop3")
    table.add_column("Volume24h")
    for c in items:
        table.add_row(
            c.ticker,
            str(c.spread_yes),
            str(c.yes_bid),
            str(c.yes_ask),
            str(c.depth_top3),
            f"{c.volume_24h:.0f}",
        )
    console.print(table)


@app.command("scan-markets")
def scan_markets_cmd(
    top: int = typer.Option(20, help="Number of markets to show"),
    min_spread: int = typer.Option(1, help="Min spread (cents)"),
    max_spread: int = typer.Option(30, help="Max spread (cents)"),
    min_trades_24h: int = typer.Option(1, help="Min trades in last 24h"),
    min_time_to_close_min: int = typer.Option(30, help="Min minutes to close"),
    status: str = typer.Option("open", help="Status filter: open/closed/settled/unopened"),
    export: Optional[str] = typer.Option(None, help="Export JSON to directory"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    scans = scan_markets_fn(
        settings,
        data_client,
        top=top,
        min_spread=min_spread,
        max_spread=max_spread,
        min_trades_24h=min_trades_24h,
        min_time_to_close_min=min_time_to_close_min,
        statuses=[status],
    )
    table = Table(title="Market Scan")
    table.add_column("Ticker")
    table.add_column("Spread")
    table.add_column("YesBid")
    table.add_column("YesAsk")
    table.add_column("Trades1h")
    table.add_column("Trades24h")
    table.add_column("Volume24h")
    for s in scans:
        table.add_row(
            s.ticker,
            str(s.spread_yes),
            str(s.best_yes_bid),
            str(s.best_yes_ask),
            str(s.trades_1h),
            str(s.trades_24h),
            f"{s.volume_24h:.0f}",
        )
    console.print(table)
    if export:
        Path(export).mkdir(parents=True, exist_ok=True)
        out = Path(export) / "market_scan.json"
        out.write_text(json.dumps([s.__dict__ for s in scans], indent=2))
        console.print(f"Wrote {out}")


@app.command("score-edge")
def score_edge_cmd(
    top: int = typer.Option(20, help="Number of markets to score"),
    min_spread: int = typer.Option(1, help="Min spread (cents)"),
    max_spread: int = typer.Option(30, help="Max spread (cents)"),
    min_trades_24h: int = typer.Option(1, help="Min trades in last 24h"),
    min_time_to_close_min: int = typer.Option(30, help="Min minutes to close"),
    status: str = typer.Option("open", help="Status filter"),
    edge_bias_cents: float = typer.Option(0.0, help="Bias implied probability (cents)"),
    export: Optional[str] = typer.Option(None, help="Export JSON to directory"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    scans = scan_markets_fn(
        settings,
        data_client,
        top=top,
        min_spread=min_spread,
        max_spread=max_spread,
        min_trades_24h=min_trades_24h,
        min_time_to_close_min=min_time_to_close_min,
        statuses=[status],
    )
    table = Table(title="Edge Scores")
    table.add_column("Ticker")
    table.add_column("Side")
    table.add_column("Price")
    table.add_column("p_model")
    table.add_column("p_implied")
    table.add_column("EV(c)")
    scores = []
    for s in scans:
        if s.best_yes_ask is None or s.best_yes_bid is None:
            continue
        mid = (s.best_yes_ask + s.best_yes_bid) / 2.0
        p_implied = implied_prob_score(int(round(mid))) or 0.5
        p_model = max(0.0, min(1.0, p_implied + edge_bias_cents / 100.0))
        score_yes_obj = score_yes(
            s.ticker,
            price_cents=int(round(s.best_yes_ask)),
            p_model=p_model,
            spread_cents=s.spread_yes or 0,
            count=1,
            maker=True,
            maker_rate=settings.execution.maker_fee_rate,
            taker_rate=settings.execution.taker_fee_rate,
            slippage_cents=0.0,
        )
        score_no_obj = score_no(
            s.ticker,
            price_cents=int(round(s.best_no_ask or (100 - s.best_yes_bid))),
            p_model=p_model,
            spread_cents=s.spread_yes or 0,
            count=1,
            maker=True,
            maker_rate=settings.execution.maker_fee_rate,
            taker_rate=settings.execution.taker_fee_rate,
            slippage_cents=0.0,
        )
        best = score_yes_obj if score_yes_obj.ev_cents >= score_no_obj.ev_cents else score_no_obj
        table.add_row(
            s.ticker,
            best.side.upper(),
            str(best.price_cents),
            f"{best.p_model:.3f}",
            f"{best.p_implied:.3f}",
            f"{best.ev_cents:.2f}",
        )
        scores.append(best.__dict__)
    console.print(table)
    if export:
        Path(export).mkdir(parents=True, exist_ok=True)
        out = Path(export) / "edge_scores.json"
        out.write_text(json.dumps(scores, indent=2))
        console.print(f"Wrote {out}")


@app.command()
def run_strategy(
    strategy: str = typer.Option("maker_ladder", help="Strategy: maker_ladder"),
    demo: bool = typer.Option(True, help="Use demo environment"),
    live: bool = typer.Option(False, help="Use live environment"),
    i_understand_risk: bool = typer.Option(False, help="Confirm live trading"),
    cycles: int = typer.Option(1, help="Number of cycles"),
    sleep: int = typer.Option(10, help="Seconds between cycles"),
    top: int = typer.Option(20, help="Number of markets to scan"),
    max_orders: int = typer.Option(5, help="Max orders per cycle"),
    levels: int = typer.Option(3, help="Ladder levels"),
    edge_bias_cents: float = typer.Option(0.0, help="Bias implied probability (cents)"),
    min_ev_cents: float = typer.Option(0.0, help="Min EV per contract (cents)"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    if live:
        demo = False
    ensure_demo_or_live(demo, live, i_understand_risk)
    if live and os.getenv("KALSHI_ARM_LIVE", "0") not in ("1", "true", "TRUE"):
        raise typer.BadParameter("Live trading requires KALSHI_ARM_LIVE=1")
    if strategy != "maker_ladder":
        raise typer.BadParameter("Only maker_ladder is supported for now")

    settings = build_settings(config)
    settings.data.env = "prod" if live else "demo"
    console.print("DEMO MODE" if not live else "LIVE MODE")
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    audit = AuditLogger(ledger, settings.log_path)
    risk = RiskManager(settings.risk)
    exec_engine = ExecutionEngine(data_client, ledger, risk, settings.execution)
    lifecycle = OrderLifecycle(exec_engine, ledger)

    for _ in range(cycles):
        scans = scan_markets_fn(settings, data_client, top=top)
        p_bias = edge_bias_cents / 100.0
        decisions = maker_ladder_cycle(
            settings,
            exec_engine,
            ledger,
            scans,
            p_model=0.5 + p_bias,
            max_orders=max_orders,
            levels=levels,
            min_ev_cents=min_ev_cents,
            lifecycle=lifecycle,
        )
        audit.log(
            "strategy",
            "maker_ladder",
            {
                "decisions": [d.score.__dict__ for d in decisions],
                "cycle_markets": len(scans),
            },
        )
        if cycles > 1:
            time.sleep(sleep)

if __name__ == "__main__":
    app()
