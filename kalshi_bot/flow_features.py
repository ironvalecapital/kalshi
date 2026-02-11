from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Optional, Tuple


@dataclass
class Trade:
    ts: datetime
    price: int
    size: int
    side: Optional[str]


class FlowFeatures:
    def __init__(self, max_trades: int = 200) -> None:
        self.trades: Deque[Trade] = deque(maxlen=max_trades)
        self.mid_history: Deque[Tuple[datetime, float]] = deque(maxlen=500)

    def update_trade(self, price: int, size: int, side: Optional[str], ts: Optional[datetime] = None) -> None:
        if ts is None:
            ts = datetime.now(timezone.utc)
        self.trades.append(Trade(ts=ts, price=price, size=size, side=side))

    def update_mid(self, mid: float) -> None:
        self.mid_history.append((datetime.now(timezone.utc), mid))

    def imbalance(self, depth_yes: int, depth_no: int) -> float:
        return (depth_yes - depth_no) / (depth_yes + depth_no + 1e-6)

    def signed_volume(self, window_sec: int = 60) -> float:
        now = datetime.now(timezone.utc)
        total = 0.0
        for t in list(self.trades):
            if (now - t.ts).total_seconds() > window_sec:
                continue
            sign = 0.0
            if t.side == "yes":
                sign = 1.0
            elif t.side == "no":
                sign = -1.0
            total += sign * t.size
        return total

    def momentum(self, window_sec: int = 30) -> float:
        now = datetime.now(timezone.utc)
        recent = [m for m in self.mid_history if (now - m[0]).total_seconds() <= window_sec]
        if len(recent) < 2:
            return 0.0
        return recent[-1][1] - recent[0][1]

    def realized_var(self, window_sec: int = 60) -> float:
        now = datetime.now(timezone.utc)
        series = [m[1] for m in self.mid_history if (now - m[0]).total_seconds() <= window_sec]
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        return sum((x - mean) ** 2 for x in series) / (len(series) - 1)
