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
from ..adapters.sportsdb import SportsDBClient
from ..adapters.football_data import FootballDataClient
from ..adapters.balldontlie import BallDontLieClient
from ..ledger import Ledger
from ..market_selector import pick_sports_candidates
from ..spread_scanner import scan_spreads
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


def _has_actionable_quotes(data_client: KalshiDataClient, ticker: str) -> bool:
    market_info = data_client.get_market(ticker)
    if market_info.get("yes_bid") is not None or market_info.get("no_bid") is not None:
        return True
    ob = data_client.get_orderbook(ticker)
    return bool(ob.get("yes")) or bool(ob.get("no"))


def _pick_best_by_spread(candidates: list) -> Optional[object]:
    if not candidates:
        return None
    def score(c):
        spread = c.spread_yes or 0
        # Prefer tighter spread and stronger recent activity.
        return (-spread * 1.0) + (c.trades_60m * 0.1)
    return max(candidates, key=score)


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
    sportsdb_client: Optional[SportsDBClient] = None
    football_client: Optional[FootballDataClient] = None
    nba_client: Optional[BallDontLieClient] = None
    mma_client: Optional[BallDontLieClient] = None
    mls_client: Optional[BallDontLieClient] = None
    ncaaf_client: Optional[BallDontLieClient] = None
    ncaab_client: Optional[BallDontLieClient] = None
    if settings.sports_external_enabled:
        sportsdb_client = SportsDBClient(api_key=settings.sportsdb_api_key)
        if settings.football_data_api_key:
            football_client = FootballDataClient(api_key=settings.football_data_api_key)
        if settings.balldontlie_api_key:
            nba_client = BallDontLieClient(api_key=settings.balldontlie_api_key, base="https://api.balldontlie.io/v1")
            mma_client = BallDontLieClient(api_key=settings.balldontlie_api_key, base="https://api.balldontlie.io/mma/v1")
            mls_client = BallDontLieClient(api_key=settings.balldontlie_api_key, base="https://api.balldontlie.io/mls/v1")
            ncaaf_client = BallDontLieClient(api_key=settings.balldontlie_api_key, base="https://api.balldontlie.io/ncaaf/v1")
            ncaab_client = BallDontLieClient(api_key=settings.balldontlie_api_key, base="https://api.balldontlie.io/ncaab/v1")

    while True:
        if exec_engine._kill_switch():
            audit.log("kill", "kill switch enabled", {})
            break

        pick = None
        if market_override:
            pick = type("Pick", (), {"ticker": market_override, "event_ticker": ""})()
        else:
            if settings.sports.use_spread_scanner:
                try:
                    spreads = scan_spreads(
                        settings,
                        data_client,
                        top=settings.sports.top_n,
                        min_spread=settings.sports.spread_scanner_min,
                        max_spread=settings.sports.spread_scanner_max,
                        status="open",
                    )
                    for s in spreads:
                        if _has_actionable_quotes(data_client, s.ticker):
                            pick = type("Pick", (), {"ticker": s.ticker, "event_ticker": ""})()
                            break
                except Exception:
                    pick = None
            auto_ticker = _auto_pick_from_summary(settings, data_client)
            if auto_ticker and pick is None and _has_actionable_quotes(data_client, auto_ticker):
                pick = type("Pick", (), {"ticker": auto_ticker, "event_ticker": ""})()
            if pick is None:
                candidates = pick_sports_candidates(settings, data_client, top_n=settings.sports.top_n)
                if not candidates:
                    audit.log("decision", "no sports candidates", {})
                    time.sleep(sleep_s)
                    if not loop_forever:
                        cycles -= 1
                        if cycles <= 0:
                            break
                    continue
                for c in sorted(candidates, key=lambda x: x.liquidity_score, reverse=True):
                    if _has_actionable_quotes(data_client, c.ticker):
                        pick = c
                        break
                if pick is None:
                    audit.log("decision", "no actionable quote candidates", {})
                    time.sleep(sleep_s)
                    if not loop_forever:
                        cycles -= 1
                        if cycles <= 0:
                            break
                    continue
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
        if settings.sports.simple_active_maker:
            if spread >= settings.sports.simple_min_spread_cents and spread <= settings.sports.max_spread_cents:
                # Simple maker: use imbalance + momentum to choose side.
                if abs(imbalance) < settings.sports.simple_imbalance_min:
                    action = "ABSTAIN"
                else:
                    action = "BID_NO" if imbalance > 0 else "BID_YES"
                edge_cents = max(0.0, (spread / 2.0) - fee)
                # Penalize if spread is widening fast.
                if spread_trend > 0.5:
                    edge_cents *= 0.5
                ev_after = edge_cents
                if ev_after < min_ev:
                    action = "ABSTAIN"
        else:
            if ev_after >= min_ev and spread <= settings.sports.max_spread_cents and fill_prob > 0.1:
                action = "BID_YES" if edge_cents >= 0 else "BID_NO"

        # Near-resolved filter: avoid extreme tails.
        if yes_ask is not None and (yes_ask <= settings.sports.avoid_price_low_cents or yes_ask >= settings.sports.avoid_price_high_cents):
            action = "ABSTAIN"

        # Longshot bias filters: avoid buying YES at extreme low prices; prefer NO at tails.
        if action == "BID_YES" and yes_ask is not None and yes_ask <= settings.sports.yes_longshot_max_cents:
            action = "ABSTAIN"
        if action == "BID_NO" and yes_ask is not None and yes_ask < settings.sports.no_tail_min_cents:
            action = "ABSTAIN"

        # Depth-aware arbitrage check (informational)
        arb_opportunity = False
        arb_spread_cents = None
        if yes_ask is not None and no_ask is not None:
            arb_spread_cents = 100 - (yes_ask + no_ask)
            if arb_spread_cents > 0 and min(depth_yes, depth_no) >= settings.sports.min_arb_depth:
                arb_opportunity = True
        if arb_opportunity and not settings.sports.allow_arb_taker:
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
            if settings.sports.simple_active_maker:
                trade_scale = max(1, int(trades_5m / 5))
                depth_scale = max(1, depth // max(1, settings.sports.depth_size_divisor))
                size = max(1, min(settings.sports.max_order_size, max(1, trade_scale + depth_scale)))
            else:
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
            if depth > 0:
                depth_cap = max(1, depth // max(1, settings.sports.depth_size_divisor))
                size = min(size, depth_cap)
            if size <= 0:
                order_result = {"status": "rejected", "reason": "kelly_size_zero"}
            else:
                ok, reason = risk.check_order(pick.ticker, size, price / 100.0)
                if ok:
                    # Ladder one additional level if spread permits.
                    ladder_levels = max(1, settings.sports.simple_ladder_levels)
                    placed = False
                    for level in range(ladder_levels):
                        price_level = price - level
                        if price_level <= 0:
                            break
                        if settings.sports.maker_only:
                            if action == "BID_YES" and yes_ask is not None and price_level >= yes_ask:
                                continue
                            if action == "BID_NO" and no_ask is not None and price_level >= no_ask:
                                continue
                        order = OrderRequest(
                            market_id=pick.ticker,
                            side="yes" if action == "BID_YES" else "no",
                            action="buy",
                            price_cents=price_level,
                            count=max(1, size // ladder_levels),
                            client_order_id=f"sports-{int(time.time())}-{level}",
                        )
                        order_result = exec_engine.place_order(order)
                        placed = placed or (order_result.get("status") == "submitted")
                        if order_result.get("status") == "submitted":
                            audit.log("order", "order submitted", {"market": pick.ticker, "side": order.side, "price": price_level, "count": order.count})
                    if not placed and order_result.get("status") != "submitted":
                        order_result = {"status": "rejected", "reason": "no_ladder_orders"}
                else:
                    order_result = {"status": "rejected", "reason": reason}

        external_meta: Dict[str, Any] = {}
        if (sportsdb_client or football_client or nba_client) and pick.title:
            # Best-effort parse "Team A vs Team B" or "Team A @ Team B"
            title = pick.title
            teams = []
            if " vs " in title:
                teams = [t.strip() for t in title.split(" vs ", 1)]
            elif " @ " in title:
                teams = [t.strip() for t in title.split(" @ ", 1)]
            if teams:
                home = teams[0]
                away = teams[1] if len(teams) > 1 else None
                try:
                    if sportsdb_client:
                        home_teams = sportsdb_client.search_teams(home)
                        away_teams = sportsdb_client.search_teams(away) if away else []
                        if home_teams:
                            team_id = home_teams[0].get("idTeam")
                            if team_id:
                                next_events = sportsdb_client.events_next(str(team_id))
                                external_meta["sportsdb_next"] = next_events[:1]
                        if away_teams and not external_meta.get("sportsdb_next"):
                            team_id = away_teams[0].get("idTeam")
                            if team_id:
                                next_events = sportsdb_client.events_next(str(team_id))
                                external_meta["sportsdb_next"] = next_events[:1]
                    if football_client and ("FC" in title or "United" in title or "City" in title):
                        fteams = football_client.search_teams(home)
                        if fteams:
                            fteam_id = fteams[0].get("id")
                            if fteam_id:
                                fmatches = football_client.matches_on_date(datetime.now(timezone.utc).date(), int(fteam_id))
                                external_meta["football_data_matches"] = [m.__dict__ for m in fmatches[:1]]
                    if nba_client and ("NBA" in title or "Lakers" in title or "Warriors" in title):
                        nteams = nba_client.search_teams(home)
                        if nteams:
                            nteam_id = nteams[0].get("id")
                            if nteam_id:
                                ngames = nba_client.games_on_date(datetime.now(timezone.utc).date(), int(nteam_id))
                                external_meta["balldontlie_games"] = [g.__dict__ for g in ngames[:1]]
                    if mma_client and ("UFC" in title or "MMA" in title):
                        # MMA API does not use team IDs; expose the upcoming events feed best-effort.
                        try:
                            mma_events = mma_client._get("/events", params={"dates[]": datetime.now(timezone.utc).date().isoformat()})
                            external_meta["balldontlie_mma_events"] = (mma_events.get("data") or [])[:1]
                        except Exception as exc:
                            external_meta["balldontlie_mma_error"] = str(exc)
                    if mls_client and ("MLS" in title or "SOCCER" in title):
                        mteams = mls_client.search_teams(home)
                        if mteams:
                            mteam_id = mteams[0].get("id")
                            if mteam_id:
                                mgames = mls_client.games_on_date(datetime.now(timezone.utc).date(), int(mteam_id))
                                external_meta["balldontlie_mls_games"] = [g.__dict__ for g in mgames[:1]]
                    if ncaaf_client and ("NCAAF" in title or "CFB" in title or "COLLEGE FOOTBALL" in title):
                        fteams = ncaaf_client.search_teams(home)
                        if fteams:
                            fteam_id = fteams[0].get("id")
                            if fteam_id:
                                fgames = ncaaf_client.games_on_date(datetime.now(timezone.utc).date(), int(fteam_id))
                                external_meta["balldontlie_ncaaf_games"] = [g.__dict__ for g in fgames[:1]]
                    if ncaab_client and ("NCAAB" in title or "COLLEGE BASKETBALL" in title):
                        bteams = ncaab_client.search_teams(home)
                        if bteams:
                            bteam_id = bteams[0].get("id")
                            if bteam_id:
                                bgames = ncaab_client.games_on_date(datetime.now(timezone.utc).date(), int(bteam_id))
                                external_meta["balldontlie_ncaab_games"] = [g.__dict__ for g in bgames[:1]]
                except Exception as exc:
                    external_meta["sportsdb_error"] = str(exc)
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
                "arb_opportunity": arb_opportunity,
                "arb_spread_cents": arb_spread_cents,
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
            "external": external_meta,
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
