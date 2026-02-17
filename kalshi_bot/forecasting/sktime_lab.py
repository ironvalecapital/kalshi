from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class SktimeModelScore:
    model: str
    mae: float
    rmse: float
    folds: int


def _series_from_array(y: Sequence[float]) -> pd.Series:
    arr = np.asarray(y, dtype=float)
    idx = pd.period_range(start="2024-01-01", periods=arr.size, freq="min")
    return pd.Series(arr, index=idx)


def compare_sktime_models(
    y: Sequence[float],
    horizon: int = 5,
    min_train_size: int = 60,
    step: int = 5,
) -> List[SktimeModelScore]:
    """
    Simple model-comparison lab for structured research:
    - NaiveForecaster
    - ARIMA
    - RandomForest reduction forecaster
    """
    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be > 0")
    if len(y) < min_train_size + horizon:
        raise ValueError("not enough points for comparison")

    from sklearn.ensemble import RandomForestRegressor
    from sktime.forecasting.arima import ARIMA
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.compose import make_reduction
    from sktime.forecasting.naive import NaiveForecaster

    rf_window = max(3, min(max(3, horizon * 2), max(3, min_train_size - horizon - 1)))
    models: Dict[str, object] = {
        "naive_last": NaiveForecaster(strategy="last"),
        "arima_111": ARIMA(order=(1, 1, 1)),
        "rf_reduction": make_reduction(
            RandomForestRegressor(n_estimators=200, random_state=7, n_jobs=-1),
            strategy="recursive",
            window_length=rf_window,
        ),
    }

    y_series = _series_from_array(y)
    scores: List[SktimeModelScore] = []

    for name, forecaster in models.items():
        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        cut = min_train_size
        while cut + horizon <= len(y_series):
            y_train = y_series.iloc[:cut]
            y_test = y_series.iloc[cut : cut + horizon]
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            try:
                forecaster.fit(y_train)
                y_pred = forecaster.predict(fh)
            except Exception:
                break
            y_true_all.extend(y_test.to_numpy(dtype=float).tolist())
            y_pred_all.extend(y_pred.to_numpy(dtype=float).tolist())
            cut += step

        if not y_true_all:
            scores.append(SktimeModelScore(model=name, mae=float("inf"), rmse=float("inf"), folds=0))
            continue
        y_true = np.asarray(y_true_all, dtype=float)
        y_pred = np.asarray(y_pred_all, dtype=float)
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        scores.append(
            SktimeModelScore(
                model=name,
                mae=mae,
                rmse=rmse,
                folds=max(0, len(y_true) // horizon),
            )
        )

    scores.sort(key=lambda s: (s.mae, s.rmse))
    return scores


def print_sktime_comparison_table(scores: Iterable[SktimeModelScore]) -> str:
    lines = ["model,mae,rmse,folds"]
    for s in scores:
        lines.append(f"{s.model},{s.mae:.6f},{s.rmse:.6f},{s.folds}")
    return "\n".join(lines)
