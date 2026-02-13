from kalshi_bot.edge_scorer import implied_probability, score_yes, score_no


def test_implied_probability_bounds():
    assert implied_probability(None) is None
    assert implied_probability(0) is None
    assert implied_probability(100) is None
    assert implied_probability(50) == 0.5


def test_score_yes_no_basic():
    score_y = score_yes(
        ticker="TEST",
        price_cents=50,
        p_model=0.6,
        spread_cents=2,
        count=1,
        maker=True,
        maker_rate=0.0175,
        taker_rate=0.07,
        slippage_cents=0,
    )
    score_n = score_no(
        ticker="TEST",
        price_cents=50,
        p_model=0.4,
        spread_cents=2,
        count=1,
        maker=True,
        maker_rate=0.0175,
        taker_rate=0.07,
        slippage_cents=0,
    )
    assert score_y.p_implied == 0.5
    assert score_n.p_implied == 0.5
