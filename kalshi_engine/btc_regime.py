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
