from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from ..adapters.nws import NWSClient, compute_daily_high_distribution
from ..config import BotSettings, merged_weather_locations
from ..data_rest import KalshiDataClient
from ..decision_models.chess_search import DecisionState, choose_action
from ..decision_report import write_decision_report
from ..ev import (
    estimate_fee_cents,
    ev_buy_no_cents,
    ev_buy_yes_cents,
    fill_probability,
    kelly_contracts,
    kelly_fraction_no,
    kelly_fraction_yes,
    spread_penalty_cents,
)
from ..execution import ExecutionEngine, OrderRequest
from ..ledger import Ledger
from ..market_picker import pick_weather_candidates
from ..risk import RiskManager


@dataclass
class MarketMapping:
    lat: float
    lon: float
    kind: str  # threshold|bucket
    threshold_f: Optional[float] = None
    low_f: Optional[float] = None
    high_f: Optional[float] = None


def parse_mapping_from_title(title: str) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    title = title or ""
    if "≥" in title:
        m = re.search(r"≥\\s*(\\d+)", title)
        if m:
            return "threshold", float(m.group(1)), None
    m = re.search(r"(\\d+)\\s*[–-]\\s*(\\d+)", title)
    if m:
        return "bucket", float(m.group(1)), float(m.group(2))
    return None


def resolve_market_mapping(
    settings: BotSettings, market: Dict[str, Any], overrides: Dict[str, Any]
) -> Optional[MarketMapping]:
    ticker = market.get("ticker", "")
    if ticker in overrides:
        ov = overrides[ticker]
        return MarketMapping(
            lat=float(ov["lat"]),
            lon=float(ov["lon"]),
            kind=str(ov.get("type", "threshold")),
            threshold_f=ov.get("threshold_f"),
            low_f=ov.get("lo"),
            high_f=ov.get("hi"),
        )
    parsed = parse_mapping_from_title(market.get("title", ""))
    if not parsed:
        return None
    city_code = None
    for code in merged_weather_locations(settings).keys():
        if code in (market.get("title", "") + market.get("subtitle", "")):
            city_code = code
            break
    if not city_code:
        city_code = settings.weather.default_city
    city = merged_weather_locations(settings).get(city_code)
    if not city:
        return None
    kind, a, b = parsed
    if kind == "threshold":
        return MarketMapping(lat=city["lat"], lon=city["lon"], kind="threshold", threshold_f=a)
    return MarketMapping(lat=city["lat"], lon=city["lon"], kind="bucket", low_f=a, high_f=b)


def implied_prices_from_orderbook(orderbook: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    # Orderbook is bids only; derive asks via complement.
    yes_bids = orderbook.get("yes", [])
    no_bids = orderbook.get("no", [])
    best_yes_bid = yes_bids[0][0] if yes_bids else None
    best_no_bid = no_bids[0][0] if no_bids else None
    best_yes_ask = 100 - best_no_bid if best_no_bid is not None else None
    best_no_ask = 100 - best_yes_bid if best_yes_bid is not None else None
    return best_yes_bid, best_yes_ask, best_no_bid, best_no_ask


def run_weather_strategy(
    settings: BotSettings,
    data_client: KalshiDataClient,
    ledger: Ledger,
    audit,
    risk: RiskManager,
    exec_engine: ExecutionEngine,
    overrides: Dict[str, Any],
    cycles: int,
    sleep_s: int,
    live: bool,
) -> None:
    if live:
        audit.log("mode", "LIVE MODE", {})
    else:
        audit.log("mode", "DEMO MODE", {})

    nws = NWSClient(user_agent=settings.weather_user_agent)
    loop_forever = cycles <= 0
    while True:
        if exec_engine._kill_switch():
            audit.log("kill", "kill switch enabled", {})
            break
        candidates = pick_weather_candidates(settings, data_client, top_n=settings.weather.top_n)
        if not candidates:
            audit.log("decision", "no market pick", {})
            time.sleep(sleep_s)
            continue
        audit.log("heartbeat", "weather cycle", {"candidates": len(candidates)})
        for cand in candidates:
            market = data_client.get_market(cand.ticker)
            mapping = resolve_market_mapping(settings, market, overrides)
            if not mapping:
                continue
            grid = nws.points(mapping.lat, mapping.lon)
            hourly = nws.hourly_forecast(grid.forecast_hourly)
            date_local = datetime.now(timezone.utc)
            mu, sigma = compute_daily_high_distribution(hourly, date_local)
            if mapping.kind == "threshold":
                p_yes = nws.daily_high_probabilities(hourly, date_local, threshold_f=float(mapping.threshold_f))
            else:
                p_yes = nws.daily_high_probabilities(hourly, date_local, bucket=(float(mapping.low_f), float(mapping.high_f)))

            yes_bid = cand.best_yes_bid
            yes_ask = cand.best_yes_ask
            no_bid = cand.best_no_bid
            no_ask = cand.best_no_ask
            if yes_bid is None or yes_ask is None:
                continue
            spread_yes = cand.spread_yes or 0
            time_to_close = (cand.close_time - datetime.now(timezone.utc)).total_seconds() / 3600.0
            fee_yes = estimate_fee_cents(1, yes_ask, True, settings.execution.maker_fee_rate, settings.execution.taker_fee_rate)
            fee_no = estimate_fee_cents(1, no_ask or 0, True, settings.execution.maker_fee_rate, settings.execution.taker_fee_rate)
            ev_yes = ev_buy_yes_cents(p_yes, yes_ask) - fee_yes - spread_penalty_cents(spread_yes)
            ev_no = ev_buy_no_cents(p_yes, no_ask or 0) - fee_no - spread_penalty_cents(spread_yes)
            depth = cand.depth_yes if action != "buy_no" else cand.depth_no
            fill_prob = fill_probability(
                spread_yes,
                cand.trades_1h,
                time_to_close,
                depth=depth,
                trades_weight=settings.weather.trades_fill_weight,
                depth_weight=settings.weather.depth_fill_weight,
            )

            state = DecisionState(
                ev_yes_cents=ev_yes,
                ev_no_cents=ev_no,
                spread_cents=spread_yes,
                volatility=1.0 / max(1.0, cand.trades_1h),
                liquidity=cand.liquidity_score,
                time_to_close_hours=time_to_close,
                risk_penalty=0.0,
            )
            action, score_val = choose_action(
                state, weights={"w_ev": 1.0, "w_spread": 0.5, "w_vol": 0.5, "w_liq": 0.2, "w_risk": 1.0}
            )

            order_result: Dict[str, Any] = {}
            size = 0
            rr_yes = (1.0 - yes_ask / 100.0) / max(0.0001, yes_ask / 100.0)
            rr_no = (1.0 - (no_ask or 1) / 100.0) / max(0.0001, (no_ask or 1) / 100.0)
            # EV gate: require fee-adjusted EV to exceed minimum
            if action in ("buy_yes", "buy_no"):
                if spread_yes > settings.weather.max_spread_cents or fill_prob < settings.weather.min_fill_prob:
                    action = "abstain"
                if time_to_close < settings.weather.min_time_to_close_hours or time_to_close > settings.weather.max_time_to_close_hours:
                    action = "abstain"
                if action == "buy_yes" and ev_yes < settings.weather.min_edge_after_fees_cents:
                    action = "abstain"
                if action == "buy_no" and ev_no < settings.weather.min_edge_after_fees_cents:
                    action = "abstain"
                if action == "buy_yes" and rr_yes < settings.weather.min_rr:
                    action = "abstain"
                if action == "buy_no" and rr_no < settings.weather.min_rr:
                    action = "abstain"
                if time_to_close <= settings.weather.regime_t2_hours:
                    if action == "buy_yes" and (ev_yes < settings.weather.near_close_min_ev_cents or rr_yes < settings.weather.near_close_min_rr):
                        action = "abstain"
                    if action == "buy_no" and (ev_no < settings.weather.near_close_min_ev_cents or rr_no < settings.weather.near_close_min_rr):
                        action = "abstain"
            if action in ("buy_yes", "buy_no"):
                if action == "buy_yes":
                    # Improve best bid by 1 cent, but keep below ask to remain maker.
                    price_cents = yes_bid + settings.weather.bid_improve_cents
                    if not settings.weather.allow_cross_spread and price_cents >= yes_ask:
                        price_cents = yes_bid
                else:
                    if no_bid is None or no_ask is None:
                        action = "abstain"
                        order_result = {"status": "rejected", "reason": "missing_no_side"}
                    else:
                        price_cents = no_bid + settings.weather.bid_improve_cents
                        if not settings.weather.allow_cross_spread and price_cents >= no_ask:
                            price_cents = no_bid
                if action in ("buy_yes", "buy_no"):
                    price_cents = max(1, price_cents)
                fee_per = estimate_fee_cents(1, price_cents, True, settings.execution.maker_fee_rate, settings.execution.taker_fee_rate)
                if action == "buy_yes":
                    kfrac = kelly_fraction_yes(p_yes, price_cents, fee_per, 0.0)
                else:
                    kfrac = kelly_fraction_no(p_yes, price_cents, fee_per, 0.0)
                size = kelly_contracts(
                    bankroll_usd=settings.execution.bankroll_usd,
                    price_cents=price_cents,
                    kelly_fraction=kfrac,
                    fractional=settings.execution.kelly_fraction,
                    fill_prob=fill_prob,
                    use_fill_prob=settings.execution.kelly_use_fill_prob,
                    max_contracts=settings.weather.max_order_size,
                )
                if size <= 0:
                    action = "abstain"
                    order_result = {"status": "rejected", "reason": "kelly_size_zero"}
                else:
                    ok, reason = risk.check_order(cand.ticker, size, price_cents / 100.0)
                    if not ok:
                        action = "abstain"
                        order_result = {"status": "rejected", "reason": reason}
            if action == "buy_yes":
                order = OrderRequest(
                    market_id=cand.ticker,
                    side="yes",
                    action="buy",
                    price_cents=price_cents,
                    count=size,
                    client_order_id=f"weather-{int(time.time())}",
                )
                order_result = exec_engine.place_order(order)
            elif action == "buy_no":
                order = OrderRequest(
                    market_id=cand.ticker,
                    side="no",
                    action="buy",
                    price_cents=price_cents,
                    count=size,
                    client_order_id=f"weather-{int(time.time())}",
                )
                order_result = exec_engine.place_order(order)
            else:
                order_result = {"status": "abstain"}

            report = {
                "inputs": {
                    "market": cand.ticker,
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "spread_yes": spread_yes,
                    "time_to_close_hours": time_to_close,
                    "depth_yes": cand.depth_yes,
                    "depth_no": cand.depth_no,
                },
                "signals": {
                    "p_yes": p_yes,
                    "ev_yes_cents": ev_yes,
                    "ev_no_cents": ev_no,
                    "fill_prob": fill_prob,
                    "search_score": score_val,
                    "rr_yes": rr_yes,
                    "rr_no": rr_no,
                    "size": size if action in ("buy_yes", "buy_no") else 0,
                },
                "chosen_action": action,
                "order_result": order_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            write_decision_report(settings, report)
            ledger.record_prediction(
                cand.ticker,
                p_yes,
                (yes_bid + yes_ask) / 2.0,
                "nws_hourly",
                {"mu": mu, "sigma": sigma},
            )
            ledger.record_decision(
                cand.ticker, "weather_high_temp", report["inputs"], report["signals"], max(ev_yes, ev_no), action, 1, {"fill_prob": fill_prob}, order_result
            )
            audit.log("decision", "weather cycle", report)
            break
        time.sleep(sleep_s)
        if not loop_forever:
            cycles -= 1
            if cycles <= 0:
                break
