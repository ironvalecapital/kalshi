from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class VariantScore:
    variant: str
    avg_ev: float
    fill_rate: float
    sharpe: float
    max_dd: float


def score_variants(rows: Iterable[Dict]) -> List[VariantScore]:
    buckets: Dict[str, List[Dict]] = {}
    for r in rows:
        buckets.setdefault(r["variant"], []).append(r)
    scored: List[VariantScore] = []
    for variant, items in buckets.items():
        if not items:
            continue
        avg_ev = sum(i.get("pnl", 0.0) for i in items) / len(items)
        fill_rate = sum(i.get("fill_rate", 0.0) for i in items) / len(items)
        sharpe = sum(i.get("sharpe", 0.0) for i in items) / len(items)
        max_dd = max(i.get("max_dd", 0.0) for i in items)
        scored.append(VariantScore(variant=variant, avg_ev=avg_ev, fill_rate=fill_rate, sharpe=sharpe, max_dd=max_dd))
    return scored


def select_champion(rows: Iterable[Dict]) -> Tuple[str, Dict]:
    scored = score_variants(rows)
    if not scored:
        return "baseline", {}
    scored.sort(key=lambda s: (s.avg_ev, s.sharpe), reverse=True)
    top = scored[0]
    return top.variant, {
        "avg_ev": top.avg_ev,
        "fill_rate": top.fill_rate,
        "sharpe": top.sharpe,
        "max_dd": top.max_dd,
    }
