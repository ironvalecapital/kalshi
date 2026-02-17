import numpy as np

from kalshi_engine.btc_regime import BTCRegimeInputs, classify_btc_regime_by_score
from kalshi_engine.bet_sizer import kelly_fraction_binary, suggest_contracts
from kalshi_engine.internal_arb import nested_probability_opportunity, time_derivative_opportunity, yes_no_bid_sum_opportunity
from kalshi_engine.monte_carlo_bayes import bayesian_probability_from_stream, monte_carlo_probability
from kalshi_engine.pipeline import build_market_decision
from kalshi_engine.sports_bayes import SportsGameState, bayesian_win_probability


def test_monte_carlo_probability_runs():
    rng = np.random.RandomState(7)

    def dist(n: int):
        return rng.normal(0, 1, n) > 0.0

    p = monte_carlo_probability(dist, n=20000)
    assert 0.45 < p < 0.55


def test_beta_stream_update_mean_bounds():
    obs = np.array([1, 1, 0, 1, 0, 1, 1])
    post = bayesian_probability_from_stream(2.0, 2.0, obs)
    assert 0.0 < post.mean < 1.0


def test_kelly_and_contracts():
    k = kelly_fraction_binary(0.6, 0.5)
    assert 0.0 < k <= 1.0
    size = suggest_contracts(bankroll_usd=100.0, price_cents=50, p_yes=0.6, fractional_kelly=0.25, max_risk_fraction=0.03)
    assert size.contracts >= 0


def test_sports_bayes_probability_bounds():
    state = SportsGameState(
        score_diff=3.0,
        time_remaining_min=10.0,
        has_possession=True,
        timeouts_team=2,
        timeouts_opp=1,
        pregame_prob=0.57,
        historical_scoring_rate_per_min=2.0,
        current_kalshi_implied=0.54,
    )
    out = bayesian_win_probability(state, n_sims=5000)
    assert 0.0 < out.model_prob < 1.0
    assert 0.0 <= out.confidence_low <= out.confidence_high <= 1.0


def test_pipeline_outputs_mode_and_snapshot():
    decision = build_market_decision(
        market_ticker="KXTEST-1",
        yes_levels=[(62, 40), (61, 30)],
        no_levels=[(37, 35), (36, 30)],
        trades=[
            {"ts": "2026-02-17T10:00:00+00:00", "yes_price": 62, "count": 10},
            {"ts": "2026-02-17T10:00:30+00:00", "yes_price": 63, "count": 12},
        ],
        bankroll=100.0,
        target_order_size=10,
        model_prob_hint=0.66,
        btc_returns=np.random.normal(0.0, 0.001, 200),
    )
    assert decision.mode in {"GTO", "EXPLOIT"}
    assert decision.snapshot["market_ticker"] == "KXTEST-1"


def test_btc_score_regime_boundaries():
    calm = classify_btc_regime_by_score(
        BTCRegimeInputs(
            realized_vol_1h=0.01,
            realized_vol_4h=0.01,
            realized_vol_24h=0.01,
            rolling_vol_avg=0.02,
            atr_ratio=0.9,
            funding_z=0.1,
            liquidation_spike_ratio=0.1,
            orderbook_imbalance=0.1,
        )
    )
    panic = classify_btc_regime_by_score(
        BTCRegimeInputs(
            realized_vol_1h=0.12,
            realized_vol_4h=0.11,
            realized_vol_24h=0.10,
            rolling_vol_avg=0.02,
            atr_ratio=2.0,
            funding_z=2.8,
            liquidation_spike_ratio=3.0,
            orderbook_imbalance=0.8,
            weekend_flag=True,
        )
    )
    assert calm.label == "calm"
    assert panic.label == "panic"


def test_yes_no_bid_sum_opportunity_detects_under_and_over():
    under = yes_no_bid_sum_opportunity("KXTEST-2", yes_bid_cents=47, no_bid_cents=50, min_edge_cents=1.0)
    over = yes_no_bid_sum_opportunity("KXTEST-2", yes_bid_cents=55, no_bid_cents=48, min_edge_cents=1.0)
    assert under is not None and under.side == "buy_both"
    assert over is not None and over.side == "sell_both"


def test_structural_constraints_detect_violations():
    nested = nested_probability_opportunity(
        parent_label="team_wins",
        parent_prob=0.58,
        child_label="team_wins_by_3",
        child_prob=0.67,
        min_gap=0.01,
    )
    temporal = time_derivative_opportunity(
        intraday_label="btc_4pm_above_x",
        intraday_prob=0.62,
        close_label="btc_close_above_x",
        close_prob=0.50,
        min_gap=0.01,
    )
    assert nested is not None and nested.reason == "nested_probability_violation"
    assert temporal is not None and temporal.reason == "time_derivative_violation"
