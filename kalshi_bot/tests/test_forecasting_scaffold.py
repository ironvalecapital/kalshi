import numpy as np

from kalshi_bot.forecasting.baselines import auto_arima_forecast
from kalshi_bot.forecasting.walk_forward import walk_forward_backtest


def test_auto_arima_forecast_shape():
    y = np.sin(np.linspace(0, 8, 80)) + 0.1 * np.random.RandomState(7).randn(80)
    out = auto_arima_forecast(y, horizon=5, seasonal=False)
    assert out.forecast.shape == (5,)
    assert len(out.order) == 3


def test_walk_forward_backtest_runs():
    y = np.linspace(1.0, 2.0, 100)

    def naive_last(train, h):
        return [train[-1]] * h

    out = walk_forward_backtest(y, naive_last, horizon=2, step=2, min_train_size=30)
    assert out.folds > 0
    assert out.mae >= 0.0
    assert out.rmse >= 0.0
