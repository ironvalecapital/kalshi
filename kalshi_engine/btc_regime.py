from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class BTCRegimeResult:
    state: int
    label: str
    transition_probs: np.ndarray
    vol_by_state: dict[int, float]

@dataclass
class BTCRegimeInputs:
    realized_vol_1h: float
    realized_vol_4h: float
    realized_vol_24h: float
    rolling_vol_avg: float
    atr_ratio: float
    funding_z: float
    liquidation_spike_ratio: float
    orderbook_imbalance: float
    weekend_flag: bool = False


@dataclass
class BTCScoreRegimeResult:
    regime_score: float
    vol_score: float
    funding_score: float
    liq_score: float
    label: str


def fit_btc_regime_detector(returns: np.ndarray, n_states: int = 3, random_state: int = 7) -> tuple[GaussianHMM, np.ndarray]:
    x = np.asarray(returns, dtype=float).reshape(-1, 1)
    if x.shape[0] < 80:
        raise ValueError("need at least 80 returns for regime fit")
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=300, random_state=random_state)
    hmm.fit(x)
    states = hmm.predict(x)
    return hmm, states


def classify_current_btc_regime(returns: np.ndarray) -> BTCRegimeResult:
    hmm, states = fit_btc_regime_detector(returns)
    current = int(states[-1])
    vol_map: dict[int, float] = {}
    for s in sorted(np.unique(states).tolist()):
        idx = states == s
        vol_map[int(s)] = float(np.std(np.asarray(returns)[idx], ddof=0))
    ordered = sorted(vol_map.items(), key=lambda kv: kv[1])
    labels = ["calm", "elevated", "crisis"]
    label_map = {st: labels[min(i, len(labels)-1)] for i, (st, _) in enumerate(ordered)}
    return BTCRegimeResult(
        state=current,
        label=label_map.get(current, "unknown"),
        transition_probs=np.asarray(hmm.transmat_[current], dtype=float),
        vol_by_state=vol_map,
    )


def classify_btc_regime_by_score(inp: BTCRegimeInputs) -> BTCScoreRegimeResult:
    """
    Rule-based regime detector aligned to live microstructure behavior:
    - Calm compression
    - Expansion / breakout
    - Panic cascade
    """
    avg_realized = max(
        1e-9,
        (float(inp.realized_vol_1h) + float(inp.realized_vol_4h) + float(inp.realized_vol_24h)) / 3.0,
    )
    vol_score = (avg_realized / max(1e-9, float(inp.rolling_vol_avg))) * max(0.5, float(inp.atr_ratio))
    funding_score = abs(float(inp.funding_z))
    liq_score = max(0.0, float(inp.liquidation_spike_ratio))

    regime_score = vol_score * 0.4 + funding_score * 0.3 + liq_score * 0.3

    # Microstructure adjustments for market stress.
    if inp.weekend_flag:
        regime_score *= 1.08
    if abs(float(inp.orderbook_imbalance)) > 0.60:
        regime_score *= 1.10

    if regime_score < 0.8:
        label = "calm"
    elif regime_score < 1.5:
        label = "expansion"
    else:
        label = "panic"

    return BTCScoreRegimeResult(
        regime_score=float(regime_score),
        vol_score=float(vol_score),
        funding_score=float(funding_score),
        liq_score=float(liq_score),
        label=label,
    )
