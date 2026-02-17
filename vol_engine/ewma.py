from __future__ import annotations

import numpy as np


def ewma_variance(returns: np.ndarray, lam: float = 0.94) -> float:
    """
    RiskMetrics-style EWMA variance forecast for next period.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise ValueError("need at least 2 return observations")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1)")

    var = float(np.var(r, ddof=0))
    for x in r:
        var = lam * var + (1.0 - lam) * float(x) * float(x)
    return max(0.0, var)


def risk_target_multiplier(
    forecast_var: float,
    target_var: float,
    floor: float = 0.25,
    cap: float = 2.0,
) -> float:
    """
    Position scaler based on target-vol / forecast-vol.
    """
    fv = max(1e-12, float(forecast_var))
    tv = max(1e-12, float(target_var))
    mult = (tv / fv) ** 0.5
    return float(np.clip(mult, floor, cap))
