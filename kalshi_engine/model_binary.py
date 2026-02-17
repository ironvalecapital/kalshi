from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def train_binary_model(
    X: pd.DataFrame,
    y: np.ndarray,
    method: str = "isotonic",
    n_estimators: int = 400,
    learning_rate: float = 0.02,
    random_state: int = 7,
) -> CalibratedClassifierCV:
    """
    Train a calibrated binary classifier for P(event=yes).
    """
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if len(X) < 50:
        raise ValueError("need at least 50 rows for stable calibration")
    base = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model = CalibratedClassifierCV(base, method=method, cv=5)
    model.fit(X, y)
    return model


def predict_yes_probability(model: CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)[:, 1]
    return np.asarray(np.clip(probs, 1e-6, 1 - 1e-6), dtype=float)


def compute_edge(model_prob: float, market_implied_prob: float) -> float:
    """
    Edge in probability points (0..1).
    """
    return float(model_prob) - float(market_implied_prob)


def model_metadata(model: Any) -> dict[str, Any]:
    return {
        "type": type(model).__name__,
        "calibration": getattr(model, "method", None),
    }
