from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .btc_regime import BTCRegimeInputs, BTCRegimeResult, BTCScoreRegimeResult, classify_btc_regime_by_score, classify_current_btc_regime
from .features import btc_tilt_score, build_llm_market_snapshot, emotion_spike_score
from .internal_arb import ArbOpportunity, find_internal_kalshi_opportunities
from .sports_bayes import SportsBayesOutput, SportsGameState, bayesian_win_probability


@dataclass
class MarketDecision:
    market_ticker: str
    mode: str
    snapshot: Dict[str, Any]
    sports: Optional[SportsBayesOutput]
    btc_regime: Optional[BTCRegimeResult]
    btc_score_regime: Optional[BTCScoreRegimeResult]
    opportunities: List[ArbOpportunity]


def decide_mode(emotion_score: float, btc_score: float, exploit_threshold: float = 1.1) -> str:
    return "EXPLOIT" if max(float(emotion_score), float(btc_score)) >= float(exploit_threshold) else "GTO"


def build_market_decision(
    *,
    market_ticker: str,
    yes_levels: list[tuple[int, int]],
    no_levels: list[tuple[int, int]],
    trades: Iterable[Dict[str, Any]],
    bankroll: float,
    target_order_size: int,
    event_ticker: Optional[str] = None,
    close_time: Optional[str] = None,
    status: Optional[str] = None,
    model_prob_hint: Optional[float] = None,
    sports_state: Optional[SportsGameState] = None,
    btc_returns: Optional[np.ndarray] = None,
    btc_inputs: Optional[BTCRegimeInputs] = None,
) -> MarketDecision:
    rows = list(trades)
    base_snapshot = build_llm_market_snapshot(
        market_ticker=market_ticker,
        event_ticker=event_ticker,
        close_time=close_time,
        status=status,
        yes_levels=yes_levels,
        no_levels=no_levels,
        trades=rows,
        model_prob=float(model_prob_hint if model_prob_hint is not None else 0.5),
        bankroll=float(bankroll),
        target_order_size=int(target_order_size),
    )

    # Sports posterior (if caller provides game state).
    sports_out: Optional[SportsBayesOutput] = None
    model_prob = float(base_snapshot.liquidity.get("implied_prob", 0.5))
    if sports_state is not None:
        sports_out = bayesian_win_probability(sports_state)
        model_prob = float(sports_out.model_prob)

    # BTC regime signal (if caller provides returns).
    btc_out: Optional[BTCRegimeResult] = None
    if btc_returns is not None and np.asarray(btc_returns).size >= 80:
        btc_out = classify_current_btc_regime(np.asarray(btc_returns, dtype=float))
    btc_score_out: Optional[BTCScoreRegimeResult] = None
    if btc_inputs is not None:
        btc_score_out = classify_btc_regime_by_score(btc_inputs)

    updated_snapshot = build_llm_market_snapshot(
        market_ticker=market_ticker,
        event_ticker=event_ticker,
        close_time=close_time,
        status=status,
        yes_levels=yes_levels,
        no_levels=no_levels,
        trades=rows,
        model_prob=model_prob,
        bankroll=float(bankroll),
        target_order_size=int(target_order_size),
    )

    # Exploit regime switch from microstructure emotion + btc tilt.
    flow = updated_snapshot.flow
    liquidity = updated_snapshot.liquidity
    emotion = emotion_spike_score(
        price_vel_1m=float(flow.get("price_velocity_1m", 0.0)),
        baseline_vol=max(1e-5, abs(float(flow.get("price_velocity_30m", 0.0))) + 1e-5),
        volume_spike_ratio=(float(flow.get("volume_velocity_1m", 0.0)) + 1e-6)
        / (float(flow.get("volume_velocity_5m", 0.0)) + 1e-6),
        spread_widening_ratio=(float(liquidity.get("spread") or 0.0) + 1.0) / 2.0,
        depth_thinning_ratio=max(0.0, 1.0 - min(1.0, (float(liquidity.get("depth_yes_5lvls", 0.0)) + float(liquidity.get("depth_no_5lvls", 0.0))) / 500.0)),
    )

    btc_score = 0.0
    if btc_score_out is not None:
        btc_score = max(
            btc_score,
            min(3.0, float(btc_score_out.regime_score)),
        )
    if btc_out is not None:
        # Crisis/elevated regimes receive larger tilt score by design.
        regime_vol = float(np.mean(list(btc_out.vol_by_state.values())) if btc_out.vol_by_state else 0.0)
        label_factor = 1.0 if btc_out.label == "crisis" else (0.6 if btc_out.label == "elevated" else 0.2)
        btc_score = btc_tilt_score(
            funding_extreme_z=label_factor,
            liquidation_spike=min(2.0, regime_vol * 100.0),
            orderbook_imbalance_ratio=abs(float(liquidity.get("imbalance", 0.0))),
            weekend_thin_liquidity_flag=1.0 if (float(liquidity.get("depth_yes_5lvls", 0.0)) + float(liquidity.get("depth_no_5lvls", 0.0))) < 120 else 0.0,
        )

    mode = decide_mode(emotion, btc_score)
    updated_snapshot.context["mode"] = mode
    updated_snapshot.flow["emotion_spike_score"] = emotion
    updated_snapshot.flow["btc_tilt_score"] = btc_score

    # Internal Kalshi arb checks.
    best_yes = updated_snapshot.liquidity.get("best_yes_bid")
    best_no = updated_snapshot.liquidity.get("best_no_bid")
    yes_ask = None if best_no is None else int(100 - int(best_no))
    no_ask = None if best_yes is None else int(100 - int(best_yes))
    opps = find_internal_kalshi_opportunities(
        market_ticker=market_ticker,
        yes_bid=None if best_yes is None else int(best_yes),
        no_bid=None if best_no is None else int(best_no),
        yes_ask=yes_ask,
        no_ask=no_ask,
        model_prob=float(updated_snapshot.model.get("model_prob", 0.5)),
        edge_threshold_cents=2.0,
    )

    return MarketDecision(
        market_ticker=market_ticker,
        mode=mode,
        snapshot=updated_snapshot.model_dump(mode="json"),
        sports=sports_out,
        btc_regime=btc_out,
        btc_score_regime=btc_score_out,
        opportunities=opps,
    )


def decision_to_dict(decision: MarketDecision) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "market_ticker": decision.market_ticker,
        "mode": decision.mode,
        "snapshot": decision.snapshot,
        "opportunities": [asdict(x) for x in decision.opportunities],
    }
    if decision.sports is not None:
        out["sports"] = asdict(decision.sports)
    if decision.btc_regime is not None:
        out["btc_regime"] = {
            "state": decision.btc_regime.state,
            "label": decision.btc_regime.label,
            "transition_probs": [float(x) for x in decision.btc_regime.transition_probs.tolist()],
            "vol_by_state": {str(k): float(v) for k, v in decision.btc_regime.vol_by_state.items()},
        }
    if decision.btc_score_regime is not None:
        out["btc_score_regime"] = asdict(decision.btc_score_regime)
    return out
