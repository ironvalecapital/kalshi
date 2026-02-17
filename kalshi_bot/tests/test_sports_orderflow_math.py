from kalshi_bot.strategies.sports_orderflow import _sigmoid, _trend_quality


def test_sigmoid_stable_at_extremes():
    assert 0.0 < _sigmoid(-10000.0) < 1.0
    assert 0.0 < _sigmoid(10000.0) < 1.0


def test_trend_quality_is_bounded():
    score = _trend_quality(
        imbalance=0.8,
        delta_mid=0.4,
        bid_mom=0.2,
        ask_mom=0.0,
        spread=2,
        spread_trend=0.1,
    )
    assert 0.0 <= score <= 1.0
