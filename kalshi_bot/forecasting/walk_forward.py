from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


ForecasterFn = Callable[[Sequence[float], int], Sequence[float]]


@dataclass
class WalkForwardResult:
    mae: float
    rmse: float
    folds: int
    y_true: np.ndarray
    y_pred: np.ndarray


def walk_forward_backtest(
    y: Sequence[float],
    forecast_fn: ForecasterFn,
    horizon: int = 1,
    step: int = 1,
    min_train_size: int = 40,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward evaluation.
    forecast_fn gets (train_series, horizon) and returns horizon predictions.
    """
    arr = np.asarray(y, dtype=float)
    if arr.size <= min_train_size + horizon:
        raise ValueError("not enough data for walk-forward")
    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be > 0")

    preds = []
    trues = []
    i = min_train_size
    while i + horizon <= arr.size:
        train = arr[:i]
        target = arr[i : i + horizon]
        pred = np.asarray(forecast_fn(train, horizon), dtype=float)
        if pred.size != horizon:
            raise ValueError("forecast_fn must return exactly `horizon` predictions")
        preds.extend(pred.tolist())
        trues.extend(target.tolist())
        i += step

    y_true = np.asarray(trues, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return WalkForwardResult(
        mae=mae,
        rmse=rmse,
        folds=max(0, len(y_true) // horizon),
        y_true=y_true,
        y_pred=y_pred,
    )
