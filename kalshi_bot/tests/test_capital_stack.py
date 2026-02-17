from datetime import datetime, timezone

from kalshi_engine.capacity_model import scan_sharpe_vs_capital
from kalshi_engine.capital_board import build_capital_health_report
from kalshi_engine.ev_forecast import TradeExpectation, simulate_forward_distribution
from kalshi_engine.rl_optimizer import ContextualBanditOptimizer, RLState, build_action_space, controls_from_action
from kalshi_engine.shadow_simulator import run_shadow_stress
from kalshi_engine.sharpe_layer import TradeRecord, allocation_weights_from_sharpe, regime_adjusted_sharpe, strategy_type_sharpe, trade_level_sharpe


def test_rl_action_space_and_controls():
    actions = build_action_space()
    assert len(actions) == 36
    agent = ContextualBanditOptimizer(seed=9)
    st = RLState("btc", "panic", 0.3, 1.4, 20.0, 0.08, 5.0, 0.5, 0.1)
    a = agent.select_action(st, explore=False)
    c = controls_from_action(a, drawdown_level=0.21)
    assert c.mode in {"GTO", "EXPLOIT"}
    assert 0.02 <= c.edge_threshold <= 0.12


def test_ev_forecast_and_capital_board():
    trades = [TradeExpectation(win_prob=0.56, price_prob=0.50, size_usd=2.0, fee_usd=0.01) for _ in range(50)]
    f = simulate_forward_distribution(trades, n_paths=200, horizon_trades=40, bankroll_start=100.0, seed=3)
    assert f.p95 >= f.p05

    pnls = [0.3, -0.2, 0.4, -0.1, 0.2] * 80
    rep = build_capital_health_report(trade_pnls=pnls, expected_edges=[0.01] * len(pnls), n_paths=500, horizon_trades=60)
    assert "capital_health" in rep
    assert 0.2 <= rep["capital_health"]["kelly_safety_factor"] <= 1.0


def test_shadow_and_capacity_and_sharpe_layers():
    s = run_shadow_stress("btc_panic", paths=120, trades_per_path=40, seed=5)
    assert 0.0 <= s.ruin_prob <= 1.0

    cap = scan_sharpe_vs_capital([20, 200, 5000], avg_depth_top5=250, avg_spread=3, volatility=0.03, edge=0.06)
    assert len(cap.points) == 3

    now = datetime.now(timezone.utc)
    rows = [
        TradeRecord(ts=now, pnl=0.2, regime="calm", strategy_type="spread"),
        TradeRecord(ts=now, pnl=-0.1, regime="panic", strategy_type="exploit"),
        TradeRecord(ts=now, pnl=0.3, regime="calm", strategy_type="arb"),
    ]
    assert isinstance(trade_level_sharpe([r.pnl for r in rows]), float)
    reg = regime_adjusted_sharpe(rows)
    typ = strategy_type_sharpe(rows)
    w = allocation_weights_from_sharpe(typ)
    assert "calm" in reg and "arb" in typ
    assert abs(sum(w.values()) - 1.0) < 1e-6
