from __future__ import annotations

import numpy as np
from arch import arch_model

from .ewma import risk_target_multiplier


def forecast_garch_variance(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
) -> float:
    """
    Fit GARCH(p, q) and return 1-step ahead variance forecast.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 30:
        raise ValueError("need at least 30 return observations for stable GARCH fit")
    # arch package works better when series scale is closer to O(1).
    scale = 1000.0
    r_scaled = r * scale
    model = arch_model(r_scaled, vol="Garch", p=p, q=q, dist="normal", mean="zero")
    res = model.fit(disp="off")
    var_scaled = float(res.forecast(horizon=1).variance.values[-1, 0])
    return max(0.0, var_scaled / (scale**2))


def position_size_scalar(
    returns: np.ndarray,
    target_var: float,
    method: str = "garch",
    ewma_lambda: float = 0.94,
) -> float:
    """
    Convenience helper:
    - forecasts next variance with GARCH or EWMA
    - converts to risk-target multiplier
    """
    m = method.lower().strip()
    if m == "garch":
        fvar = forecast_garch_variance(returns)
    elif m == "ewma":
        from .ewma import ewma_variance

        fvar = ewma_variance(returns, lam=ewma_lambda)
    else:
        raise ValueError("method must be 'garch' or 'ewma'")
    return risk_target_multiplier(forecast_var=fvar, target_var=target_var)
