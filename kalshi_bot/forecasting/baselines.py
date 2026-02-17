from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class AutoArimaResult:
    forecast: np.ndarray
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None


def auto_arima_forecast(
    y: Sequence[float],
    horizon: int,
    seasonal: bool = False,
    m: int = 1,
) -> AutoArimaResult:
    """
    Fit pmdarima auto_arima and return horizon forecast.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    arr = np.asarray(y, dtype=float)
    if arr.size < 16:
        raise ValueError("need at least 16 observations for stable auto_arima")

    from pmdarima import auto_arima

    model = auto_arima(
        arr,
        seasonal=seasonal,
        m=max(1, int(m)),
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        with_intercept=True,
    )
    pred = np.asarray(model.predict(n_periods=horizon), dtype=float)
    seasonal_order = None
    if seasonal:
        seasonal_order = tuple(model.seasonal_order)
    return AutoArimaResult(
        forecast=pred,
        order=tuple(model.order),
        seasonal_order=seasonal_order,
    )
