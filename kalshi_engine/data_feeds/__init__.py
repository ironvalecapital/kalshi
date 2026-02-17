from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WeatherSignal:
    temperature_f: float
    precip_mm: float
    wind_mph: float


@dataclass
class CpiSignal:
    cpi_yoy: float
    core_cpi_yoy: float
    energy_index_yoy: float


@dataclass
class SportsSignal:
    elo_diff: float
    injury_impact: float
    home_advantage: float


@dataclass
class RatesSignal:
    fed_funds_implied: float
    yield_2y: float
    yield_10y: float


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert mixed API payload values into numeric features.
    Non-numeric/null values are dropped.
    """
    out: Dict[str, float] = {}
    for k, v in payload.items():
        try:
            if v is None:
                continue
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out
