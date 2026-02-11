from kalshi_bot.tools.breakeven import breakeven_yes, breakeven_no


def test_breakeven_yes():
    r = breakeven_yes(52, 100, True, 0)
    assert r["p_break_even"] >= r["p_market"]
    assert r["edge_cents"] >= 0


def test_breakeven_no():
    r = breakeven_no(48, 50, False, 1)
    assert r["p_break_even_no"] >= r["p_market_no"]
    assert r["edge_cents"] >= 0
