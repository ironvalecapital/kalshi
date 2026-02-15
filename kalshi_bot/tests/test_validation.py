from kalshi_bot.analytics.validation import brier_score, edge_ttest, reliability_curve, walk_forward_windows


def test_brier_score_basic():
    assert brier_score([0.5, 0.9], [1, 1]) > 0


def test_edge_ttest_positive_mean():
    stats = edge_ttest([0.02, 0.03, -0.01, 0.04, 0.02] * 50)
    assert stats["n"] == 250
    assert stats["mean"] > 0
    assert 0 <= stats["p_value_one_sided"] <= 1


def test_reliability_curve_has_bins():
    rows = reliability_curve([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], bins=4)
    assert len(rows) >= 2


def test_walk_forward_windows():
    windows = walk_forward_windows(total=100, train=40, test=10, step=10)
    assert windows[0] == (0, 40, 40, 50)
    assert windows[-1][3] <= 100

