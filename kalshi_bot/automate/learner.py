from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import BotSettings
from ..data_rest import KalshiDataClient
from ..decision_report import write_decision_report
from ..ev import estimate_fee_cents, ev_buy_yes_cents, ev_buy_no_cents, fill_probability, spread_penalty_cents
from ..ledger import Ledger
from ..living_files import append_known_failure, update_playbook_weather, write_memory_pack, write_live_flag, write_operating_rules
from ..market_selector import pick_sports_candidates
from ..fee_model import fee_cents
from ..market_picker import pick_weather_candidates
from ..strategies.weather_high_temp import resolve_market_mapping
from ..adapters.nws import NWSClient, compute_daily_high_distribution
from .promotion import select_champion


@dataclass
class VariantResult:
    variant: str
    p_hat: float
    implied: float
    ev_yes: float
    ev_no: float
    fill_prob: float


def _implied_from_prices(yes_bid: Optional[int], yes_ask: Optional[int]) -> float:
    if yes_bid is None or yes_ask is None:
        return 0.5
    return (yes_bid + yes_ask) / 200.0


def evaluate_variants(
    settings: BotSettings,
    candidate,
    nws: Optional[NWSClient],
    mapping,
) -> List[VariantResult]:
    yes_bid = candidate.best_yes_bid
    yes_ask = candidate.best_yes_ask
    no_ask = candidate.best_no_ask
    implied = _implied_from_prices(yes_bid, yes_ask)

    p_forecast = implied
    if nws and mapping:
        hourly = nws.hourly_forecast(nws.points(mapping.lat, mapping.lon).forecast_hourly)
        date_local = datetime.now(timezone.utc)
        compute_daily_high_distribution(hourly, date_local)
        if mapping.kind == "threshold":
            p_forecast = nws.daily_high_probabilities(hourly, date_local, threshold_f=float(mapping.threshold_f))
        else:
            p_forecast = nws.daily_high_probabilities(hourly, date_local, bucket=(float(mapping.low_f), float(mapping.high_f)))

    # Microstructure variant: small adjustment by depth imbalance and trade activity.
    depth_yes = candidate.depth_yes
    depth_no = candidate.depth_no
    imbalance = (depth_yes - depth_no) / max(1.0, depth_yes + depth_no)
    p_micro = max(0.01, min(0.99, implied + 0.02 * imbalance))

    # Time-to-close variant: increase conservatism as close nears.
    now = datetime.now(timezone.utc)
    time_to_close = (candidate.close_time - now).total_seconds() / 3600.0
    p_time = implied
    if time_to_close <= settings.weather.regime_t2_hours:
        p_time = max(0.01, min(0.99, implied + 0.03 * (1 if imbalance > 0 else -1)))

    results = []
    for name, p_hat in [("forecast", p_forecast), ("micro", p_micro), ("time", p_time)]:
        fee_yes = estimate_fee_cents(1, yes_ask or 0, True, settings.execution.maker_fee_rate, settings.execution.taker_fee_rate)
        fee_no = estimate_fee_cents(1, no_ask or 0, True, settings.execution.maker_fee_rate, settings.execution.taker_fee_rate)
        ev_yes = ev_buy_yes_cents(p_hat, yes_ask or 50) - fee_yes - spread_penalty_cents(candidate.spread_yes or 0)
        ev_no = ev_buy_no_cents(p_hat, no_ask or 50) - fee_no - spread_penalty_cents(candidate.spread_yes or 0)
        fill_prob = fill_probability(
            candidate.spread_yes or 0,
            candidate.trades_1h,
            time_to_close,
            depth=candidate.depth_yes + candidate.depth_no,
            trades_weight=settings.weather.trades_fill_weight,
            depth_weight=settings.weather.depth_fill_weight,
        )
        results.append(VariantResult(name, p_hat, implied, ev_yes, ev_no, fill_prob))
    return results


def run_learn(
    settings: BotSettings,
    data_client: KalshiDataClient,
    ledger: Ledger,
    audit,
    lane: str,
    loop: bool,
    interval_sec: int,
) -> None:
    write_operating_rules()
    nws = None
    if lane == "weather":
        try:
            nws = NWSClient(user_agent=settings.weather_user_agent)
        except Exception as exc:
            append_known_failure(f"NWS init failed: {exc}")

    while True:
        try:
            if lane == "weather":
                candidates = pick_weather_candidates(settings, data_client, top_n=settings.weather.top_n)
            else:
                candidates = pick_sports_candidates(settings, data_client, top_n=settings.sports.top_n)
            if not candidates:
                audit.log("learn", "no candidates", {})
                time.sleep(interval_sec)
                if not loop:
                    break
                continue
            results_rows = []
            top_markets = []
            max_cycles = settings.weather.max_trades_per_cycle if lane == "weather" else settings.sports.top_n
            for cand in candidates[: max_cycles]:
                top_markets.append(cand.ticker)
                if lane == "weather":
                    market = data_client.get_market(cand.ticker)
                    mapping = resolve_market_mapping(settings, market, settings.weather.market_overrides or {})
                    for res in evaluate_variants(settings, cand, nws, mapping):
                        metrics = {
                            "implied": res.implied,
                            "p_hat": res.p_hat,
                            "ev_yes": res.ev_yes,
                            "ev_no": res.ev_no,
                            "fill_rate": res.fill_prob,
                            "sharpe": res.ev_yes / max(0.01, abs(res.ev_no)),
                            "max_dd": min(0.0, res.ev_yes),
                        }
                        ledger.record_experiment(
                            cand.ticker,
                            res.variant,
                            {"lane": lane},
                            metrics,
                            pnl=max(res.ev_yes, res.ev_no),
                            sharpe=metrics["sharpe"],
                            max_dd=metrics["max_dd"],
                            fill_rate=res.fill_prob,
                            fee_drag=0.0,
                            spread_drag=cand.spread_yes or 0.0,
                        )
                        results_rows.append(
                            {
                                "variant": res.variant,
                                "pnl": max(res.ev_yes, res.ev_no),
                                "fill_rate": res.fill_prob,
                                "sharpe": metrics["sharpe"],
                                "max_dd": metrics["max_dd"],
                            }
                        )
                else:
                    yes_ask = cand.best_yes_ask or 50
                    implied = (cand.best_yes_bid + yes_ask) / 200.0 if cand.best_yes_bid is not None else 0.5
                    # Variant A: spread capture maker
                    fee = fee_cents(1, yes_ask, maker=True)
                    ev_spread = max(0.0, (cand.spread_yes or 0) / 2.0 - fee)
                    # Variant B: imbalance tilt
                    imbalance = 0.0
                    if cand.depth_top3:
                        imbalance = 0.1
                    ev_imbalance = ev_spread + (0.5 if imbalance > 0 else 0.0)
                    for name, ev in [("spread_maker", ev_spread), ("imbalance_maker", ev_imbalance)]:
                        metrics = {
                            "implied": implied,
                            "p_hat": implied,
                            "ev_yes": ev,
                            "ev_no": -ev,
                            "fill_rate": 0.2,
                            "sharpe": ev / 0.01,
                            "max_dd": min(0.0, ev),
                        }
                        ledger.record_experiment(
                            cand.ticker,
                            name,
                            {"lane": lane},
                            metrics,
                            pnl=ev,
                            sharpe=metrics["sharpe"],
                            max_dd=metrics["max_dd"],
                            fill_rate=metrics["fill_rate"],
                            fee_drag=fee,
                            spread_drag=cand.spread_yes or 0.0,
                        )
                        results_rows.append(
                            {
                                "variant": name,
                                "pnl": ev,
                                "fill_rate": metrics["fill_rate"],
                                "sharpe": metrics["sharpe"],
                                "max_dd": metrics["max_dd"],
                            }
                        )
                write_decision_report(settings, {"lane": lane, "market": cand.ticker, "ts": datetime.now(timezone.utc).isoformat()})

            champion, metrics = select_champion(results_rows)
            thresholds = {
                "entry_edge_pp": settings.weather.entry_edge_pp if lane == "weather" else None,
                "min_edge_after_fees_cents": settings.weather.min_edge_after_fees_cents if lane == "weather" else None,
                "max_spread_cents": settings.weather.max_spread_cents if lane == "weather" else settings.sports.resolved_max_spread_cents(),
            }
            if lane == "weather":
                update_playbook_weather(thresholds)
            write_memory_pack(
                lane=lane,
                top_markets=top_markets,
                chosen_variant=champion,
                thresholds=thresholds,
                recent_metrics=metrics,
                kill_switch=False,
            )
            live_ready = metrics.get("avg_ev", 0.0) > 0 and metrics.get("fill_rate", 0.0) > 0.1
            write_live_flag(live_ready, metrics)
            audit.log("learn", "cycle complete", {"champion": champion, "metrics": metrics})
        except Exception as exc:
            append_known_failure(f"learn loop error: {exc}")
            audit.log("learn", "error", {"error": str(exc)})

        if not loop:
            break
        time.sleep(interval_sec)
