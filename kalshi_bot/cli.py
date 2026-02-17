from __future__ import annotations

import asyncio
import csv
from collections import Counter
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import numpy as np
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
from .market_selector import diagnose_sports_markets, pick_sports_candidates
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
from .backtest.monte_carlo import MonteCarloConfig, run_growth_simulation, style_configs
from .analytics.validation import edge_ttest
from .forecasting import compare_sktime_models, print_sktime_comparison_table
from .watchlist import build_watchlist
from .watchlist_server import serve_watchlist
from .alerts import send_telegram
from .models.live_repricing import LiveState, monte_carlo_win_probability, win_probability
from kalshi_engine.layer_backtest import run_layer_comparison
from kalshi_engine.rl_optimizer import ContextualBanditOptimizer, RLState, controls_from_action, ev_ratio_reward, risk_adjusted_reward
from kalshi_engine.ev_forecast import bootstrap_pnl_distribution
from kalshi_engine.capital_board import build_capital_health_report
from kalshi_engine.shadow_simulator import SCENARIOS, run_shadow_stress
from kalshi_engine.capacity_model import scan_sharpe_vs_capital
from kalshi_engine.institutional_report import export_institutional_report

app = typer.Typer(add_completion=False)
console = Console()


def _ticker_family_match(ticker: str, family: str) -> bool:
    fam = (family or "all").lower().strip()
    if fam == "all":
        return True
    t = (ticker or "").upper()
    rules = {
        "sports": ("NBA", "NFL", "NHL", "MLB", "NCAA", "MATCH", "GAME", "SPORT"),
        "crypto": ("BTC", "ETH", "CRYPTO"),
        "finance": ("CPI", "FED", "RATE", "INFLATION", "FINANCE", "SPX", "DJIA", "NASDAQ"),
    }
    return any(tok in t for tok in rules.get(fam, ()))


def _normalize_family(family: str) -> str:
    fam = (family or "all").lower().strip()
    if fam == "auto":
        return "all"
    if fam not in {"all", "sports", "crypto", "finance"}:
        raise typer.BadParameter("--family must be one of: auto, all, sports, crypto, finance")
    return fam


def _queue_fill_prob(queue_ahead: int, trades_5m: int, spread: float, horizon_sec: int = 60) -> float:
    trade_rate_sec = max(0.0, float(trades_5m) / 300.0)
    aggressiveness = max(0.2, 1.4 - 0.08 * max(0.0, float(spread)))
    expected = trade_rate_sec * float(horizon_sec) * aggressiveness
    q = max(0.0, float(queue_ahead))
    return max(0.0, min(1.0, 1.0 - math.exp(-expected / (q + 1.0))))


def build_settings(config_path: Optional[str]) -> BotSettings:
    settings = load_config(config_path)
    limits = tier_to_limits(settings.rate_limit.tier)
    settings.rate_limit.read_per_sec = limits["read"]
    settings.rate_limit.write_per_sec = limits["write"]
    settings.rate_limit.burst = limits["burst"]
    # Institutional hard cap: keep Kelly multiplier <= 15% until formal review says otherwise.
    settings.execution.kelly_fraction = min(settings.execution.kelly_fraction, 0.15)
    return settings


@app.command("engine-compare")
def engine_compare(
    steps: int = typer.Option(2500, help="Simulation steps"),
    bankroll: float = typer.Option(100.0, help="Starting bankroll"),
    seed: int = typer.Option(7, help="Random seed"),
):
    """
    Compare layer performance:
    pure prob vs +exploit vs +microstructure vs +internal-arb.
    """
    rows = run_layer_comparison(seed=seed, steps=steps, bankroll_start=bankroll)
    table = Table(title="Kalshi Engine Layer Backtest")
    table.add_column("Layer")
    table.add_column("Trades", justify="right")
    table.add_column("WinRate", justify="right")
    table.add_column("AvgEdge", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("MaxDD", justify="right")
    for r in rows:
        table.add_row(
            r.name,
            str(r.trades),
            f"{r.win_rate:.3f}",
            f"{r.avg_edge:.4f}",
            f"{r.pnl:.2f}",
            f"{r.max_drawdown:.3f}",
        )
    console.print(table)


@app.command("bandit-train")
def bandit_train(
    episodes: int = typer.Option(3000, help="Synthetic training episodes"),
    seed: int = typer.Option(7, help="Random seed"),
    log_sar: bool = typer.Option(True, help="Log state-action-reward tuples to CSV"),
):
    """
    Train contextual bandit for mode/kelly/edge/execution controls.
    """
    rng = np.random.default_rng(seed)
    agent = ContextualBanditOptimizer(seed=seed)
    sar_rows: List[Dict[str, Any]] = []
    for _ in range(episodes):
        state = RLState(
            market_type="btc" if rng.random() < 0.45 else "sports",
            volatility_regime="panic" if rng.random() < 0.20 else ("expansion" if rng.random() < 0.45 else "calm"),
            liquidity_score=float(rng.uniform(0.1, 1.0)),
            emotion_score=float(rng.uniform(0.0, 2.0)),
            time_to_expiry_min=float(rng.uniform(5, 360)),
            edge_size=float(rng.uniform(0.0, 0.15)),
            spread_width=float(rng.uniform(1, 12)),
            depth_imbalance=float(rng.uniform(-1, 1)),
            drawdown_level=float(rng.uniform(0.0, 0.3)),
        )
        idx = agent.select_action_idx(state, explore=True)
        action = agent.actions[idx]
        controls = controls_from_action(action, state.drawdown_level)
        base_edge = state.edge_size - controls.edge_threshold
        style_penalty = 0.05 if (controls.execution_style == "aggressive" and state.liquidity_score < 0.35) else 0.0
        mode_bonus = 0.03 if (controls.mode == "EXPLOIT" and state.emotion_score > 1.2) else 0.0
        realized = base_edge + mode_bonus - style_penalty + float(rng.normal(0.0, 0.03))
        var_pen = (0.02 + max(0.0, state.spread_width - 4.0) * 0.01)
        dd_pen = state.drawdown_level * 0.15
        reward = 0.5 * risk_adjusted_reward(realized, var_pen, dd_pen) + 0.5 * ev_ratio_reward(realized, max(1e-4, base_edge + 0.01))
        agent.update(state, idx, reward)
        if log_sar:
            sar_rows.append(
                {
                    "market_type": state.market_type,
                    "volatility_regime": state.volatility_regime,
                    "liquidity_score": state.liquidity_score,
                    "emotion_score": state.emotion_score,
                    "time_to_expiry_min": state.time_to_expiry_min,
                    "edge_size": state.edge_size,
                    "spread_width": state.spread_width,
                    "depth_imbalance": state.depth_imbalance,
                    "drawdown_level": state.drawdown_level,
                    "action_mode": action.mode,
                    "action_kelly_multiplier": action.kelly_multiplier,
                    "action_edge_threshold": action.edge_threshold,
                    "action_execution_style": action.execution_style,
                    "reward": reward,
                }
            )

    sample = RLState(
        market_type="btc",
        volatility_regime="panic",
        liquidity_score=0.32,
        emotion_score=1.6,
        time_to_expiry_min=25.0,
        edge_size=0.09,
        spread_width=6.0,
        depth_imbalance=0.58,
        drawdown_level=0.08,
    )
    best = agent.select_action(sample, explore=False)
    tuned = controls_from_action(best, sample.drawdown_level)
    console.print(
        {
            "episodes": episodes,
            "recommended_action": {
                "mode": best.mode,
                "kelly_multiplier": best.kelly_multiplier,
                "edge_threshold": best.edge_threshold,
                "execution_style": best.execution_style,
            },
            "live_controls": {
                "mode": tuned.mode,
                "edge_threshold": tuned.edge_threshold,
                "kelly_fraction": tuned.kelly_fraction,
                "execution_style": tuned.execution_style,
            },
        }
    )
    if log_sar and sar_rows:
        out_dir = Path("runs/rl")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sar_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sar_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sar_rows)
        console.print({"sar_log": str(out_path)})


@app.command("ev-forward")
def ev_forward(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    paths: int = typer.Option(3000, help="Monte Carlo paths"),
    horizon: int = typer.Option(250, help="Trades horizon"),
):
    """
    Forward EV and PnL distribution forecast from historical trade PnLs.
    """
    settings = build_settings(config)
    ledger = Ledger(settings.db_path)
    pnls = ledger.get_trade_pnls(limit=5000)
    if not pnls:
        console.print("No trade PnLs in ledger yet.")
        return
    out = bootstrap_pnl_distribution(pnls, n_paths=paths, horizon_trades=horizon, seed=7)
    table = Table(title="Forward EV Forecast (Bootstrap PnL Distribution)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Mean PnL", f"{out.mean_pnl:.2f}")
    table.add_row("Std PnL", f"{out.stdev_pnl:.2f}")
    table.add_row("P05", f"{out.p05:.2f}")
    table.add_row("Median", f"{out.p50:.2f}")
    table.add_row("P95", f"{out.p95:.2f}")
    table.add_row("CVaR 5%", f"{out.cvar_05:.2f}")
    table.add_row("Ruin Prob", f"{out.ruin_prob:.3f}")
    console.print(table)


@app.command("capital-board")
def capital_board(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    """
    Daily capital health snapshot with Kelly safety recommendation.
    """
    settings = build_settings(config)
    ledger = Ledger(settings.db_path)
    pnls = ledger.get_trade_pnls(limit=5000)
    if not pnls:
        console.print("No trade PnLs in ledger yet.")
        return

    con = sqlite3.connect(settings.db_path)
    cur = con.cursor()
    cur.execute("SELECT expected_edge FROM decisions WHERE expected_edge IS NOT NULL ORDER BY id DESC LIMIT 5000")
    edges = [float(r[0]) / 100.0 for r in cur.fetchall() if r[0] is not None]
    con.close()

    report = build_capital_health_report(trade_pnls=pnls, expected_edges=edges, n_paths=10000, horizon_trades=300)
    h = report["capital_health"]
    table = Table(title="Capital Allocation Board")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Expected Return 30d", f"{h['expected_return_30d']:.2f}")
    table.add_row("Worst 5% Drawdown", f"{h['worst_5pct_drawdown']:.2f}")
    table.add_row("Negative Month Prob", f"{h['negative_month_prob']:.3f}")
    table.add_row("Risk of Ruin", f"{h['risk_of_ruin']:.3f}")
    table.add_row("EV Ratio", f"{h['ev_ratio']:.3f}")
    table.add_row("Sharpe-like", f"{h['sharpe_like']:.3f}")
    table.add_row("Kelly Safety Factor", f"{h['kelly_safety_factor']:.3f}")
    table.add_row("Global Kelly Rec", f"{report['global_kelly_recommendation']:.3f}")
    table.add_row("Stress Level", str(report["stress_level"]))
    console.print(table)


@app.command("shadow-stress")
def shadow_stress(
    scenario: str = typer.Option("btc_panic", help=f"Scenario: {','.join(SCENARIOS.keys())}"),
    paths: int = typer.Option(1000, help="Monte Carlo paths"),
    trades_per_path: int = typer.Option(150, help="Trades per path"),
):
    """
    Nightly emotional stress simulation for Sports/BTC.
    """
    if scenario not in SCENARIOS:
        raise typer.BadParameter(f"scenario must be one of: {', '.join(SCENARIOS.keys())}")
    out = run_shadow_stress(scenario_key=scenario, paths=paths, trades_per_path=trades_per_path)
    console.print(out.__dict__)


@app.command("capacity-scan")
def capacity_scan(
    edge: float = typer.Option(0.06, help="Base edge (probability points)"),
    depth: float = typer.Option(250.0, help="Average top5 depth"),
    spread: float = typer.Option(3.0, help="Average spread cents"),
    vol: float = typer.Option(0.03, help="Volatility estimate"),
):
    """
    Liquidity capacity model and Sharpe-vs-capital scan.
    """
    caps = [20.0, 200.0, 5000.0, 50000.0, 250000.0, 1000000.0]
    out = scan_sharpe_vs_capital(caps, avg_depth_top5=depth, avg_spread=spread, volatility=vol, edge=edge)
    table = Table(title="Sharpe vs Capital Curve")
    table.add_column("Capital", justify="right")
    table.add_column("OrderSize", justify="right")
    table.add_column("Slippage", justify="right")
    table.add_column("EffEdge", justify="right")
    table.add_column("SharpeLike", justify="right")
    for p in out.points:
        table.add_row(
            f"{p.capital:.0f}",
            f"{p.order_size:.2f}",
            f"{p.slippage:.4f}",
            f"{p.effective_edge:.4f}",
            f"{p.sharpe_like:.3f}",
        )
    console.print(table)
    console.print({"recommended_max_capital": out.recommended_max_capital})


@app.command("investor-report")
def investor_report(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    out_dir: str = typer.Option("runs/investor_report", help="Output directory"),
):
    """
    Export institutional summary report (MD + CSV + optional PDF).
    """
    settings = build_settings(config)
    ledger = Ledger(settings.db_path)
    pnls = ledger.get_trade_pnls(limit=5000)
    health = build_capital_health_report(trade_pnls=pnls, expected_edges=[], n_paths=4000, horizon_trades=200)
    payload = {
        "strategy_overview": {
            "venue": "Kalshi",
            "focus": ["sports", "btc"],
            "alpha_streams": ["probability", "exploit", "microstructure", "internal_arb", "market_making"],
        },
        "sharpe_history": {"sharpe_like": health["capital_health"]["sharpe_like"]},
        "ev_ratio": health["capital_health"]["ev_ratio"],
        "drawdown": {
            "worst_5pct_drawdown": health["capital_health"]["worst_5pct_drawdown"],
            "stress_level": health["stress_level"],
        },
        "risk_of_ruin": health["capital_health"]["risk_of_ruin"],
        "capacity_limits": {"global_kelly_recommendation": health["global_kelly_recommendation"]},
    }
    files = export_institutional_report(payload, out_dir=out_dir)
    console.print(files)


@app.command("weekly-summary")
def weekly_summary(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    """
    Weekly regime summary from recent decisions/audit rows.
    """
    settings = build_settings(config)
    con = sqlite3.connect(settings.db_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT signals_json FROM decisions
        WHERE ts >= datetime('now', '-7 day')
        ORDER BY id DESC
        LIMIT 5000
        """
    )
    rows = [r[0] for r in cur.fetchall() if r and r[0]]
    con.close()
    regime_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    for raw in rows:
        try:
            obj = json.loads(raw)
            regime = str(obj.get("regime") or "unknown")
            mode = str(obj.get("mode") or "unknown")
            regime_counts[regime] += 1
            mode_counts[mode] += 1
        except Exception:
            continue
    console.print(
        {
            "weekly_decisions": len(rows),
            "regime_distribution": dict(regime_counts),
            "mode_distribution": dict(mode_counts),
        }
    )


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


@app.command("scrape-bball-ref")
def scrape_bball_ref(
    season: int = typer.Option(2026, help="NBA season year (e.g. 2026)"),
    outdir: str = typer.Option("data/basketball_reference", help="Output directory"),
):
    """
    Scrape Basketball-Reference team profiles for institutional feature research.
    Includes:
    - efficiency + pace
    - home/away bias
    - scoring splits
    - Q4 scoring patterns
    - red-zone conversion proxy (2P FG%)
    """
    try:
        from .adapters.basketball_reference import BasketballReferenceClient
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "Basketball scrape requires optional dependency `beautifulsoup4`. "
            "Install with: pip install beautifulsoup4"
        ) from exc
    client = BasketballReferenceClient()
    try:
        rows = client.scrape_team_profiles(season=season)
    finally:
        client.close()
    if not rows:
        console.print("No rows scraped.")
        raise typer.Exit(1)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out / f"nba_team_profiles_{season}_{ts}.csv"
    json_path = out / f"nba_team_profiles_{season}_{ts}.json"

    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))

    table = Table(title=f"Basketball-Reference Team Profiles ({season})")
    table.add_column("Team")
    table.add_column("OffRtg")
    table.add_column("DefRtg")
    table.add_column("Pace")
    table.add_column("HomeWin%")
    table.add_column("AwayWin%")
    table.add_column("Q4 Net")
    for r in rows[:15]:
        table.add_row(
            str(r.get("team", "")),
            "" if r.get("off_rtg") is None else f"{r['off_rtg']:.2f}",
            "" if r.get("def_rtg") is None else f"{r['def_rtg']:.2f}",
            "" if r.get("pace") is None else f"{r['pace']:.2f}",
            "" if r.get("home_win_pct") is None else f"{100*r['home_win_pct']:.1f}%",
            "" if r.get("away_win_pct") is None else f"{100*r['away_win_pct']:.1f}%",
            "" if r.get("q4_net") is None else f"{r['q4_net']:.2f}",
        )
    console.print(table)
    console.print(f"Wrote: {csv_path}")
    console.print(f"Wrote: {json_path}")


@app.command("live-reprice")
def live_reprice(
    score_diff: int = typer.Option(..., help="Team score - opponent score"),
    time_remaining_sec: int = typer.Option(..., help="Seconds remaining"),
    possession: int = typer.Option(0, help="+1 team, -1 opponent, 0 unknown"),
    efficiency_edge: float = typer.Option(0.0, help="PPP edge (team - opp)"),
    pace: float = typer.Option(98.0, help="Possessions per 48 minutes"),
    market_prob: Optional[float] = typer.Option(None, help="Current market implied prob (0-1)"),
    simulations: int = typer.Option(2000, help="Monte Carlo paths"),
):
    fast_p = win_probability(score_diff, time_remaining_sec, possession, efficiency_edge, pace)
    mc_p = monte_carlo_win_probability(
        LiveState(
            score_diff=score_diff,
            time_remaining_sec=time_remaining_sec,
            possession=possession,
            efficiency_edge=efficiency_edge,
            pace=pace,
        ),
        simulations=simulations,
    )
    table = Table(title="Live Probability Repricing")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Fast model P(win)", f"{fast_p:.4f}")
    table.add_row("Monte Carlo P(win)", f"{mc_p:.4f}")
    if market_prob is not None:
        edge_pp = (mc_p - market_prob) * 100.0
        table.add_row("Market implied", f"{market_prob:.4f}")
        table.add_row("Edge (pp)", f"{edge_pp:.2f}")
    console.print(table)


@app.command()
def hot_tickers(
    top: int = typer.Option(20, help="Top N tickers by recent trade count"),
    limit: int = typer.Option(200, help="Trades page size"),
    max_pages: int = typer.Option(10, help="Max pages to read"),
    family: str = typer.Option("auto", help="auto|all|sports|crypto|finance"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    family = _normalize_family(family)

    counts: Counter[str] = Counter()
    cursor: Optional[str] = None
    pages = 0
    while pages < max_pages:
        resp = data_client.get_trades(limit=limit, cursor=cursor)
        trades = resp.get("trades", []) or []
        if not trades:
            break
        for t in trades:
            ticker = str(t.get("ticker", "")).upper()
            if not ticker:
                continue
            if not _ticker_family_match(ticker, family):
                continue
            if "MULTIGAMEEXTENDED" in ticker or "QUICKSETTLE" in ticker:
                continue
            counts[ticker] += 1
        cursor = resp.get("cursor")
        pages += 1
        if not cursor:
            break

    if not counts:
        console.print("No traded tickers found for current filters.")
        raise typer.Exit(0)

    table = Table(title="Hot Tickers (Live Tape)")
    table.add_column("Ticker")
    table.add_column("Status")
    table.add_column("Trades")
    table.add_column("YesBid")
    table.add_column("YesAsk")
    table.add_column("Spread")
    for ticker, n in counts.most_common(top):
        m = data_client.get_market(ticker)
        status = str(m.get("status", "")).lower()
        if status and status not in {"open", "unopened"}:
            continue
        yes_bid = m.get("yes_bid")
        no_bid = m.get("no_bid")
        if yes_bid is None and no_bid is None:
            ob = data_client.get_orderbook(ticker)
            yes_levels = ob.get("yes", []) or []
            no_levels = ob.get("no", []) or []
            yes_bid = yes_levels[0][0] if yes_levels else None
            no_bid = no_levels[0][0] if no_levels else None
        yes_ask = m.get("yes_ask")
        if yes_ask is None and no_bid is not None:
            yes_ask = 100 - no_bid
        spread = (yes_ask - yes_bid) if yes_ask is not None and yes_bid is not None else None
        table.add_row(
            ticker,
            status or "",
            str(n),
            "" if yes_bid is None else str(yes_bid),
            "" if yes_ask is None else str(yes_ask),
            "" if spread is None else str(spread),
        )
    console.print(table)


@app.command()
def hot_edge(
    top: int = typer.Option(20, help="Top N edge-ranked tickers from live tape"),
    limit: int = typer.Option(300, help="Trades page size"),
    max_pages: int = typer.Option(8, help="Max pages to read"),
    family: str = typer.Option("auto", help="auto|all|sports|crypto|finance"),
    count: int = typer.Option(5, help="Contracts for fee/EV estimation"),
    notify: bool = typer.Option(False, help="Send top scan summary to Telegram"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    family = _normalize_family(family)

    # Gather a recent tape window once.
    tape: list[dict] = []
    cursor: Optional[str] = None
    pages = 0
    while pages < max_pages:
        resp = data_client.get_trades(limit=limit, cursor=cursor)
        chunk = resp.get("trades", []) or []
        if not chunk:
            break
        tape.extend(chunk)
        cursor = resp.get("cursor")
        pages += 1
        if not cursor:
            break

    def build_rows_for_family(fam: str) -> list[dict[str, Any]]:
        counts: Counter[str] = Counter()
        for tr in tape:
            ticker = str(tr.get("ticker", "")).upper()
            if not ticker:
                continue
            if not _ticker_family_match(ticker, fam):
                continue
            if "MULTIGAMEEXTENDED" in ticker or "QUICKSETTLE" in ticker:
                continue
            counts[ticker] += 1
        rows_local: list[dict[str, Any]] = []
        for ticker, n in counts.most_common(max(top * 3, 40)):
            m = data_client.get_market(ticker)
            status = str(m.get("status", "")).lower()
            if status and status not in {"open", "unopened"}:
                continue
            ob = data_client.get_orderbook(ticker)
            yes_bids = ob.get("yes", []) or []
            no_bids = ob.get("no", []) or []
            yes_bid = yes_bids[0][0] if yes_bids else m.get("yes_bid")
            no_bid = no_bids[0][0] if no_bids else m.get("no_bid")
            yes_ask = 100 - no_bid if no_bid is not None else m.get("yes_ask")
            if yes_bid is None or yes_ask is None:
                continue
            spread = yes_ask - yes_bid
            if spread < 0:
                continue
            depth_yes = sum(int(level[1]) for level in yes_bids[:3]) if yes_bids else 0
            signed = 0.0
            trades_5m = 0
            for tr in tape:
                if str(tr.get("ticker", "")).upper() != ticker:
                    continue
                side = str(tr.get("taker_side") or tr.get("side") or "").lower()
                size = float(tr.get("count") or 0)
                if side == "yes":
                    signed += size
                elif side == "no":
                    signed -= size
                trades_5m += 1
            p_implied = yes_ask / 100.0
            tape_tilt = max(-0.05, min(0.05, signed / max(1.0, float(n) * 20.0)))
            p_model = max(0.01, min(0.99, p_implied + tape_tilt))
            edge_cents = (p_model - p_implied) * 100.0
            fee_total = estimate_fee_cents(
                count=count,
                price_cents=int(yes_bid + 1),
                maker=True,
                maker_rate=settings.execution.maker_fee_rate,
                taker_rate=settings.execution.taker_fee_rate,
            )
            fee_per = fee_total / max(1, count)
            queue_ahead = int(yes_bids[0][1]) if yes_bids else max(1, depth_yes)
            fill_prob = _queue_fill_prob(queue_ahead=queue_ahead, trades_5m=trades_5m, spread=spread, horizon_sec=60)
            ev_cents = (edge_cents - fee_per) * fill_prob
            rows_local.append(
                {
                    "ticker": ticker,
                    "family": fam,
                    "status": status or "",
                    "trades": n,
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "spread": spread,
                    "edge": edge_cents,
                    "fill_prob": fill_prob,
                    "ev_exec": ev_cents,
                }
            )
        return rows_local

    rows = build_rows_for_family(family)
    if not rows and family != "all":
        fallback_families = ["all", "crypto", "finance", "sports"]
        for fam in fallback_families:
            if fam == family:
                continue
            rows = build_rows_for_family(fam)
            if rows:
                console.print(f"No actionable rows for family={family}. Falling back to family={fam}.")
                break

    rows.sort(key=lambda r: r["ev_exec"], reverse=True)
    if not rows:
        # Fallback debug view to help user pick a real ticker quickly.
        counts: Counter[str] = Counter()
        for tr in tape:
            ticker = str(tr.get("ticker", "")).upper()
            if not ticker or "MULTIGAMEEXTENDED" in ticker or "QUICKSETTLE" in ticker:
                continue
            if _ticker_family_match(ticker, family):
                counts[ticker] += 1
        if counts:
            debug = Table(title="Hot Tape (No Actionable Quotes Yet)")
            debug.add_column("Ticker")
            debug.add_column("Status")
            debug.add_column("Trades")
            debug.add_column("YesBid")
            debug.add_column("NoBid")
            shown = 0
            for ticker, n in counts.most_common(top):
                m = data_client.get_market(ticker)
                status = str(m.get("status", "")).lower()
                if status and status not in {"open", "unopened"}:
                    continue
                ob = data_client.get_orderbook(ticker)
                yes_levels = ob.get("yes", []) or []
                no_levels = ob.get("no", []) or []
                yes_bid = yes_levels[0][0] if yes_levels else m.get("yes_bid")
                no_bid = no_levels[0][0] if no_levels else m.get("no_bid")
                debug.add_row(
                    ticker,
                    status or "",
                    str(n),
                    "" if yes_bid is None else str(yes_bid),
                    "" if no_bid is None else str(no_bid),
                )
                shown += 1
                if shown >= top:
                    break
            console.print(debug)
        console.print("No edge-ranked rows yet (likely no actionable bid/ask on recent tape for this family).")
        console.print("Tip: run `python -m kalshi_bot.cli hot-tickers --top 50 --family auto --config configs/example.yaml` and watch one live ticker.")
        raise typer.Exit(0)
    table = Table(title="Hot Edge Rank (Tape + Queue + Fees)")
    table.add_column("Ticker")
    table.add_column("Family")
    table.add_column("Status")
    table.add_column("Trades")
    table.add_column("YesBid")
    table.add_column("YesAsk")
    table.add_column("Spread")
    table.add_column("Edge(c)")
    table.add_column("FillProb")
    table.add_column("EVexec(c)")
    shown = 0
    for r in rows:
        table.add_row(
            r["ticker"],
            r["family"],
            r["status"],
            str(r["trades"]),
            str(r["yes_bid"]),
            str(r["yes_ask"]),
            str(r["spread"]),
            f"{r['edge']:.2f}",
            f"{r['fill_prob']:.2f}",
            f"{r['ev_exec']:.2f}",
        )
        shown += 1
        if shown >= top:
            break
    console.print(table)
    if notify:
        lines = []
        for r in rows[: min(top, 10)]:
            lines.append(
                f"{r['ticker']} | {r['family']} | tr={r['trades']} | spr={r['spread']} | ev={r['ev_exec']:.2f}"
            )
        send_telegram("Kalshi hot-edge scan", {"top": lines, "family": family})


@app.command()
def scan_notify_loop(
    interval: int = typer.Option(180, help="Seconds between scans"),
    top: int = typer.Option(20, help="Top N rows"),
    family: str = typer.Option("auto", help="auto|all|sports|crypto|finance"),
    limit: int = typer.Option(300, help="Trades page size"),
    max_pages: int = typer.Option(8, help="Max pages to read"),
    count: int = typer.Option(5, help="Contracts for fee/EV estimate"),
    notify_empty: bool = typer.Option(False, help="Send Telegram even when no actionable rows"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    while True:
        try:
            hot_edge(
                top=top,
                limit=limit,
                max_pages=max_pages,
                family=family,
                count=count,
                notify=True,
                config=config,
            )
        except typer.Exit:
            if notify_empty:
                send_telegram("Kalshi scan: no actionable rows", {"family": family})
        time.sleep(max(15, interval))


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
    family: str = typer.Option("auto", help="Market family: auto|all|sports|crypto|finance"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    # Keep interactive picker fast under rate limits.
    settings.sports.orderbook_probe_limit = min(settings.sports.orderbook_probe_limit, max(20, top * 2))
    settings.sports.selector_workers = 1
    _, data_client = build_clients(settings)
    family = _normalize_family(family)
    candidates = pick_sports_candidates(settings, data_client, top_n=top, family=family)
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
def pick_markets(
    top: int = typer.Option(30, help="Top N"),
    family: str = typer.Option("auto", help="Market family: auto|all|sports|crypto|finance"),
    include_excluded: bool = typer.Option(False, help="Include excluded/family-mismatch rows"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    settings.sports.orderbook_probe_limit = min(settings.sports.orderbook_probe_limit, max(30, top * 2))
    settings.sports.selector_workers = 1
    _, data_client = build_clients(settings)
    rows = diagnose_sports_markets(
        settings,
        data_client,
        top_n=top,
        family=family,
        include_excluded=include_excluded,
    )
    if not rows:
        console.print("No markets returned.")
        raise typer.Exit(0)
    table = Table(title="Market Diagnostics")
    table.add_column("Ticker")
    table.add_column("Reason")
    table.add_column("YesBid")
    table.add_column("YesAsk")
    table.add_column("NoBid")
    table.add_column("NoAsk")
    table.add_column("Spread")
    table.add_column("Score")
    for r in rows:
        table.add_row(
            r.ticker,
            r.reason,
            "" if r.best_yes_bid is None else str(r.best_yes_bid),
            "" if r.best_yes_ask is None else str(r.best_yes_ask),
            "" if r.best_no_bid is None else str(r.best_no_bid),
            "" if r.best_no_ask is None else str(r.best_no_ask),
            "" if r.spread_yes is None else str(r.spread_yes),
            f"{r.liquidity_score:.2f}",
        )
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
    family: str = typer.Option("auto", help="Market family: auto|all|sports|crypto|finance"),
    poll_sec: int = typer.Option(2, help="Polling seconds"),
    auto_rotate: bool = typer.Option(True, help="Rotate to next hot ticker if no quotes"),
    rotate_after_empty: int = typer.Option(8, help="Rotate after N empty polls"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    _, data_client = build_clients(settings)
    family = _normalize_family(family)
    flow = FlowFeatures()
    current_market = market
    empty_polls = 0
    zero_depth_polls = 0
    zero_depth_warn_every = 10

    def _derive_book_snapshot(ticker: str) -> Dict[str, Any]:
        ob = data_client.get_orderbook(ticker)
        book = ob.get("orderbook") if isinstance(ob, dict) and isinstance(ob.get("orderbook"), dict) else ob
        yes_raw = book.get("yes", []) or book.get("yes_bids", [])
        no_raw = book.get("no", []) or book.get("no_bids", [])

        def _norm(levels: List[Any]) -> List[List[int]]:
            out: List[List[int]] = []
            for lvl in levels:
                if isinstance(lvl, list) and len(lvl) >= 2:
                    out.append([int(lvl[0]), int(lvl[1])])
                elif isinstance(lvl, dict) and lvl.get("price") is not None and lvl.get("size") is not None:
                    out.append([int(lvl["price"]), int(lvl["size"])])
            return out

        yes_levels = _norm(yes_raw)
        no_levels = _norm(no_raw)
        best_yes_bid = yes_levels[0][0] if yes_levels else None
        best_no_bid = no_levels[0][0] if no_levels else None
        best_yes_ask = 100 - best_no_bid if best_no_bid is not None else None
        depth_yes = sum(int(level[1]) for level in yes_levels[:3]) if yes_levels else 0
        depth_no = sum(int(level[1]) for level in no_levels[:3]) if no_levels else 0

        # REST summary fallback when orderbook is empty.
        if best_yes_bid is None and best_no_bid is None:
            m = data_client.get_market(ticker)
            best_yes_bid = m.get("yes_bid")
            best_no_bid = m.get("no_bid")
            if best_yes_ask is None and best_no_bid is not None:
                best_yes_ask = 100 - best_no_bid
        spread = (best_yes_ask - best_yes_bid) if best_yes_ask is not None and best_yes_bid is not None else None
        return {
            "best_yes_bid": best_yes_bid,
            "best_yes_ask": best_yes_ask,
            "best_no_bid": best_no_bid,
            "spread": spread,
            "depth_yes": depth_yes,
            "depth_no": depth_no,
        }

    def _next_hot_ticker(current: str) -> Optional[str]:
        resp = data_client.get_trades(limit=300)
        counts: Counter[str] = Counter()
        for tr in resp.get("trades", []) or []:
            ticker = str(tr.get("ticker", "")).upper()
            if not ticker or ticker == current.upper():
                continue
            if not _ticker_family_match(ticker, family):
                continue
            counts[ticker] += 1
        for ticker, _ in counts.most_common(25):
            snap = _derive_book_snapshot(ticker)
            if snap["best_yes_bid"] is not None or snap["best_no_bid"] is not None:
                return ticker
        return None

    while True:
        snap = _derive_book_snapshot(current_market)
        best_yes = snap["best_yes_bid"]
        best_yes_ask = snap["best_yes_ask"]
        if best_yes is not None and best_yes_ask is not None:
            flow.update_mid((best_yes + best_yes_ask) / 2)
        depth_yes = snap["depth_yes"]
        depth_no = snap["depth_no"]
        console.print(
            {
                "market": current_market,
                "best_yes": best_yes,
                "best_yes_ask": best_yes_ask,
                "spread": snap["spread"],
                "imbalance": flow.imbalance(depth_yes, depth_no),
                "momentum_30s": flow.momentum(30),
            }
        )

        if depth_yes == 0 and depth_no == 0:
            zero_depth_polls += 1
            if zero_depth_polls % zero_depth_warn_every == 0:
                console.print(
                    {
                        "market": current_market,
                        "warning": "no depth on both sides (data likely stale/thin)",
                        "zero_depth_polls": zero_depth_polls,
                    }
                )
        else:
            zero_depth_polls = 0

        if best_yes is None and snap["best_no_bid"] is None:
            empty_polls += 1
            if auto_rotate and empty_polls >= rotate_after_empty:
                next_ticker = _next_hot_ticker(current_market)
                if next_ticker and next_ticker != current_market:
                    console.print(f"Rotating to hot ticker: {next_ticker}")
                    current_market = next_ticker
                    empty_polls = 0
        else:
            empty_polls = 0
        time.sleep(max(1, poll_sec))


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
    arm_live: bool = typer.Option(False, help="Set KALSHI_ARM_LIVE=1 for this run"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    cycles: int = typer.Option(1, help="Number of cycles"),
    sleep: int = typer.Option(10, help="Seconds between cycles"),
    market: Optional[str] = typer.Option(None, help="Override market ticker"),
    markets: Optional[str] = typer.Option(None, help="Comma-separated market tickers to run in rotation"),
    family: str = typer.Option("auto", help="Market family: auto|all|sports|crypto|finance"),
    pyramid_off: bool = typer.Option(False, help="Disable add-to-winner pyramiding for this run"),
    stdout_events: bool = typer.Option(True, help="Print decision/order events to stdout"),
):
    if live:
        demo = False
    if arm_live:
        os.environ["KALSHI_ARM_LIVE"] = "1"
    if stdout_events:
        os.environ["KALSHI_STDOUT_EVENTS"] = "1"
    ensure_demo_or_live(demo, live, i_understand_risk)
    if live and os.getenv("KALSHI_ARM_LIVE", "0") not in ("1", "true", "TRUE"):
        raise typer.BadParameter(
            "Live trading requires KALSHI_ARM_LIVE=1. "
            "Run `export KALSHI_ARM_LIVE=1` first or pass `--arm-live`."
        )
    family = _normalize_family(family)
    market_list = [t.strip() for t in (markets or "").split(",") if t.strip()]
    if market and market not in market_list:
        market_list.append(market)
    # Keep live loop responsive under API throttling.
    settings = build_settings(config)
    if pyramid_off:
        settings.sports.pyramid_winners_enabled = False
    settings.sports.top_n = min(settings.sports.top_n, 60)
    settings.sports.max_scan_markets = min(settings.sports.max_scan_markets, 400)
    settings.sports.orderbook_probe_limit = min(settings.sports.orderbook_probe_limit, 80)
    settings.data.env = "prod" if live else "demo"
    console.print("DEMO MODE" if not live else "LIVE MODE")
    _, data_client = build_clients(settings)
    ledger = Ledger(settings.db_path)
    audit = AuditLogger(ledger, settings.log_path)
    risk = RiskManager(settings.risk)
    exec_engine = ExecutionEngine(data_client, ledger, risk, settings.execution)
    run_sports_strategy(
        settings,
        data_client,
        ledger,
        audit,
        risk,
        exec_engine,
        cycles,
        sleep,
        live,
        market_override=market,
        market_overrides=market_list or None,
        family=family,
    )


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


@app.command("limiter-report")
def limiter_report(
    last: int = typer.Option(500, help="Inspect the last N decision events from audit log"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    """
    Summarize why sports decisions abstained and print tuning hints.
    """
    settings = build_settings(config)
    log_path = Path(settings.log_path)
    if not log_path.exists():
        console.print(f"Audit log not found: {log_path}")
        raise typer.Exit(1)

    raw_lines = log_path.read_text().splitlines()
    limiter_counts: Counter[str] = Counter()
    total_decisions = 0
    total_abstains = 0

    for line in reversed(raw_lines):
        if total_decisions >= max(1, last):
            break
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event_type") != "decision" or row.get("message") != "sports orderflow":
            continue
        total_decisions += 1
        ctx = row.get("context", {}) if isinstance(row.get("context"), dict) else {}
        action = str(ctx.get("action") or "")
        limiter = str(ctx.get("decision_limiter") or "unknown")
        if action == "ABSTAIN":
            total_abstains += 1
            limiter_counts[limiter] += 1

    if total_decisions == 0:
        console.print("No sports decision events found in audit log yet.")
        raise typer.Exit(0)

    console.print(
        f"Decisions inspected: {total_decisions} | Abstains: {total_abstains} "
        f"({(100.0 * total_abstains / total_decisions):.1f}%)"
    )

    hint_map = {
        "ev_below_min": "Lower `sports.min_ev_cents` or reduce fee drag with tighter entries.",
        "ev_or_fill_or_spread_gate": "Lower EV gate or fill gate; increase `max_spread_pct` if intentional.",
        "imbalance_too_low": "Lower `sports.simple_imbalance_min` to permit weaker signals.",
        "near_resolved_tail_filter": "Widen `avoid_price_low_cents`/`avoid_price_high_cents` only if desired.",
        "longshot_yes_filter": "Raise `yes_longshot_max_cents` if you want more longshot YES entries.",
        "no_tail_filter": "Lower `no_tail_min_cents` if you want more NO entries outside extreme tails.",
        "illiquid_penalty": "Reduce `illiquid_ev_penalty_cents` or lower liquidity minimums.",
        "arb_taker_disabled": "Set `allow_arb_taker: true` to permit taker arb fills.",
        "no_signal": "Increase scan set or lower signal thresholds; market may just be quiet.",
    }

    table = Table(title="Decision Limiters")
    table.add_column("Limiter")
    table.add_column("Count")
    table.add_column("Share")
    table.add_column("Hint")
    for limiter, count in limiter_counts.most_common():
        share = f"{(100.0 * count / max(1, total_abstains)):.1f}%"
        table.add_row(limiter, str(count), share, hint_map.get(limiter, "Review log details for this limiter."))
    console.print(table)


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


@app.command("ops-report")
def ops_report(
    lookback: int = typer.Option(500, help="Number of recent trades to evaluate"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    ledger = Ledger(settings.db_path)
    pnls = ledger.get_trade_pnls(limit=max(50, lookback))
    stats = edge_ttest(pnls)

    table = Table(title="Ironvale Ops Report")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Recent trades", str(stats["n"]))
    table.add_row("Mean pnl/trade", f"{stats['mean']:.4f}")
    table.add_row("t-stat", f"{stats['t_stat']:.3f}")
    table.add_row("p-value (one-sided)", f"{stats['p_value_one_sided']:.4f}")
    table.add_row("Kelly cap", f"{100 * min(0.15, settings.execution.kelly_fraction):.1f}%")
    table.add_row("Per-market exposure cap", f"${settings.risk.max_position_per_market_usd:.2f}")
    table.add_row("Per-order max notional", f"${settings.risk.max_notional_usd:.2f}")
    table.add_row("Rule: no changes before 500 trades", "PASS" if stats["n"] >= 500 else "LOCKED")
    table.add_row("Rule: significance p < 0.05", "PASS" if stats["p_value_one_sided"] < 0.05 else "FAIL")
    console.print(table)

    manual = Path("docs/IRONVALE_PROP_DESK_OPERATING_MANUAL.md")
    if manual.exists():
        console.print(f"Operating manual: {manual}")


@app.command("simulate-growth")
def simulate_growth(
    style: str = typer.Option("adaptive", help="aggressive|institutional|adaptive"),
    bankroll: float = typer.Option(5.0, help="Starting bankroll in USD"),
    years: int = typer.Option(5, help="Years to simulate"),
    trades_per_year: int = typer.Option(500, help="Trades per year"),
    win_prob: float = typer.Option(0.58, help="Per-trade win probability"),
    edge_up: float = typer.Option(0.03, help="Fractional gain on winning trade"),
    paths: int = typer.Option(10000, help="Monte Carlo paths"),
):
    presets = style_configs()
    key = style.strip().lower()
    if key not in presets:
        raise typer.BadParameter("style must be one of: aggressive, institutional, adaptive")
    cfg = presets[key]
    cfg.bankroll_start = bankroll
    cfg.years = years
    cfg.trades_per_year = trades_per_year
    cfg.win_prob = win_prob
    cfg.edge_up = edge_up
    cfg.paths = paths
    stats = run_growth_simulation(cfg)

    table = Table(title=f"Growth Simulation ({key})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Start bankroll", f"${cfg.bankroll_start:.2f}")
    table.add_row("Trades/year", str(cfg.trades_per_year))
    table.add_row("Years", str(cfg.years))
    table.add_row("Median final", f"${stats['median_final']:.2f}")
    table.add_row("P25 final", f"${stats['p25_final']:.2f}")
    table.add_row("P75 final", f"${stats['p75_final']:.2f}")
    table.add_row("P95 final", f"${stats['p95_final']:.2f}")
    table.add_row("Median max DD", f"{100 * stats['median_max_dd']:.1f}%")
    table.add_row("Ruin probability", f"{100 * stats['ruin_prob']:.1f}%")
    table.add_row("70%+ DD probability", f"{100 * stats['deep_dd_prob']:.1f}%")
    console.print(table)


@app.command("forecast-lab")
def forecast_lab(
    market: str = typer.Option(..., help="Kalshi ticker to evaluate"),
    source: str = typer.Option("ticks", help="ticks|trades"),
    horizon: int = typer.Option(3, help="Forecast horizon"),
    min_train: int = typer.Option(60, help="Minimum training observations"),
    step: int = typer.Option(5, help="Walk-forward step"),
    limit: int = typer.Option(600, help="Max rows loaded from DB"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
):
    settings = build_settings(config)
    db_path = settings.db_path
    if source not in {"ticks", "trades"}:
        raise typer.BadParameter("source must be one of: ticks, trades")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if source == "ticks":
            cur.execute(
                """
                SELECT mid FROM ticks
                WHERE market_id = ? AND mid IS NOT NULL
                ORDER BY ts DESC
                LIMIT ?
                """,
                (market, max(50, limit)),
            )
        else:
            cur.execute(
                """
                SELECT price FROM market_trades
                WHERE market_id = ? AND price IS NOT NULL
                ORDER BY ts DESC
                LIMIT ?
                """,
                (market, max(50, limit)),
            )
        vals = [float(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()

    if len(vals) < (min_train + horizon + 5):
        raise typer.BadParameter(
            f"not enough {source} data for {market}. found={len(vals)} "
            f"need_at_least={min_train + horizon + 5}"
        )

    series = list(reversed(vals))
    scores = compare_sktime_models(
        series,
        horizon=horizon,
        min_train_size=min_train,
        step=step,
    )

    table = Table(title=f"Forecast Lab ({market})")
    table.add_column("Model")
    table.add_column("MAE")
    table.add_column("RMSE")
    table.add_column("Folds")
    for s in scores:
        table.add_row(s.model, f"{s.mae:.6f}", f"{s.rmse:.6f}", str(s.folds))
    console.print(table)
    console.print(print_sktime_comparison_table(scores))


if __name__ == "__main__":
    app()
