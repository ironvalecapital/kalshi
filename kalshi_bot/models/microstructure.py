from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MicrostructureResult:
    size_multiplier: float
    required_edge_cents: float
    seconds_to_close: Optional[float]


def adjust_for_event_time(
    close_time: Optional[datetime],
    base_required_edge_cents: float,
) -> MicrostructureResult:
    if close_time is None:
        return MicrostructureResult(size_multiplier=1.0, required_edge_cents=base_required_edge_cents, seconds_to_close=None)
    now = datetime.now(timezone.utc)
    seconds = (close_time - now).total_seconds()
    if seconds <= 0:
        return MicrostructureResult(size_multiplier=0.0, required_edge_cents=base_required_edge_cents * 2, seconds_to_close=seconds)
    if seconds < 3600:
        return MicrostructureResult(size_multiplier=0.3, required_edge_cents=base_required_edge_cents * 2, seconds_to_close=seconds)
    if seconds < 6 * 3600:
        return MicrostructureResult(size_multiplier=0.6, required_edge_cents=base_required_edge_cents * 1.5, seconds_to_close=seconds)
    return MicrostructureResult(size_multiplier=1.0, required_edge_cents=base_required_edge_cents, seconds_to_close=seconds)
