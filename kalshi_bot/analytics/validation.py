from __future__ import annotations

import math
import statistics
from typing import Iterable, List, Sequence, Tuple


def brier_score(preds: Sequence[float], outcomes: Sequence[int]) -> float:
    if not preds or not outcomes or len(preds) != len(outcomes):
        return 0.0
    err = [(float(p) - float(o)) ** 2 for p, o in zip(preds, outcomes)]
    return float(sum(err) / len(err))


def reliability_curve(preds: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> List[dict]:
    if not preds or not outcomes or len(preds) != len(outcomes):
        return []
    b = max(2, int(bins))
    rows: List[dict] = []
    for i in range(b):
        lo = i / b
        hi = (i + 1) / b
        idx = [k for k, p in enumerate(preds) if lo <= float(p) < hi or (i == b - 1 and float(p) <= hi)]
        if not idx:
            continue
        p_hat = sum(float(preds[k]) for k in idx) / len(idx)
        p_real = sum(float(outcomes[k]) for k in idx) / len(idx)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": len(idx),
                "predicted": p_hat,
                "realized": p_real,
                "gap": p_hat - p_real,
            }
        )
    return rows


def edge_ttest(returns: Sequence[float]) -> dict:
    vals = [float(x) for x in returns]
    n = len(vals)
    if n < 2:
        return {"n": n, "mean": 0.0, "std": 0.0, "t_stat": 0.0, "p_value_one_sided": 1.0}
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals)
    if sd <= 1e-12:
        return {"n": n, "mean": mu, "std": sd, "t_stat": 0.0, "p_value_one_sided": 1.0 if mu <= 0 else 0.0}
    t = mu / (sd / math.sqrt(n))
    # Normal approximation for one-sided p-value (sufficient for n>=30).
    p = 0.5 * math.erfc(t / math.sqrt(2.0))
    return {"n": n, "mean": mu, "std": sd, "t_stat": t, "p_value_one_sided": max(0.0, min(1.0, p))}


def walk_forward_windows(total: int, train: int, test: int, step: int | None = None) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    n = max(0, int(total))
    tr = max(1, int(train))
    te = max(1, int(test))
    st = te if step is None else max(1, int(step))
    start = 0
    while start + tr + te <= n:
        out.append((start, start + tr, start + tr, start + tr + te))
        start += st
    return out


def significant_edge(returns: Sequence[float], alpha: float = 0.05, min_n: int = 100) -> bool:
    stats = edge_ttest(returns)
    return bool(stats["n"] >= min_n and stats["mean"] > 0 and stats["p_value_one_sided"] < alpha)

