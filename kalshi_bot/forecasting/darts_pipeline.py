from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class DartsEnsembleResult:
    forecast: np.ndarray
    model_name: str


def darts_simple_ensemble_forecast(
    y: Sequence[float],
    horizon: int,
    freq: str = "1min",
) -> DartsEnsembleResult:
    """
    Lightweight Darts scaffold for fast prototyping:
    - ARIMA
    - ExponentialSmoothing
    Uses simple average ensemble.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    arr = np.asarray(y, dtype=float)
    if arr.size < 24:
        raise ValueError("need at least 24 observations for Darts ensemble")

    from darts import TimeSeries
    from darts.models import ARIMA, ExponentialSmoothing

    idx = pd.date_range("2024-01-01", periods=arr.size, freq=freq)
    ts = TimeSeries.from_times_and_values(idx, arr)

    m1 = ARIMA()
    m2 = ExponentialSmoothing()
    m1.fit(ts)
    m2.fit(ts)

    f1 = m1.predict(horizon).values().reshape(-1)
    f2 = m2.predict(horizon).values().reshape(-1)
    forecast = 0.5 * (f1 + f2)
    return DartsEnsembleResult(forecast=np.asarray(forecast, dtype=float), model_name="ARIMA+ETS")
