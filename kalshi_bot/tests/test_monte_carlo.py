from kalshi_bot.backtest.monte_carlo import MonteCarloConfig, run_growth_simulation


def test_growth_simulation_outputs_bounds():
    cfg = MonteCarloConfig(paths=200, years=1, trades_per_year=50, seed=11)
    stats = run_growth_simulation(cfg)
    assert stats["median_final"] >= 0
    assert 0 <= stats["ruin_prob"] <= 1
    assert 0 <= stats["deep_dd_prob"] <= 1

