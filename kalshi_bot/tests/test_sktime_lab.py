import numpy as np

from kalshi_bot.forecasting.sktime_lab import compare_sktime_models


def test_compare_sktime_models_runs():
    rng = np.random.RandomState(11)
    t = np.linspace(0, 12, 180)
    y = np.sin(t) + 0.15 * rng.randn(180)
    scores = compare_sktime_models(y, horizon=3, min_train_size=60, step=6)
    assert len(scores) == 3
    assert scores[0].mae >= 0.0
