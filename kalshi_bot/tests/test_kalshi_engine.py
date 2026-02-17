import numpy as np

from kalshi_engine.bet_sizer import kelly_fraction_binary, suggest_contracts
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
