from kalshi_bot.strategies.sports_orderflow import _sigmoid


def test_sigmoid_stable_at_extremes():
    assert 0.0 < _sigmoid(-10000.0) < 1.0
    assert 0.0 < _sigmoid(10000.0) < 1.0
