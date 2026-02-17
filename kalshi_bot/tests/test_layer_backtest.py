from kalshi_engine.layer_backtest import run_layer_comparison


def test_layer_comparison_runs_and_returns_four_layers():
    rows = run_layer_comparison(seed=11, steps=400, bankroll_start=100.0)
    assert len(rows) == 4
    names = {r.name for r in rows}
    assert names == {"pure_prob", "prob_plus_exploit", "prob_exploit_micro", "full_internal_arb"}
    for r in rows:
        assert r.trades >= 0
        assert 0.0 <= r.win_rate <= 1.0
        assert r.max_drawdown >= 0.0
