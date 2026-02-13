from kalshi_bot.automate.promotion import select_champion


def test_select_champion():
    rows = [
        {"variant": "a", "pnl": 1.0, "fill_rate": 0.2, "sharpe": 0.5, "max_dd": -0.1},
        {"variant": "b", "pnl": 0.5, "fill_rate": 0.2, "sharpe": 1.0, "max_dd": -0.1},
    ]
    champion, metrics = select_champion(rows)
    assert champion == "a"
    assert "avg_ev" in metrics
