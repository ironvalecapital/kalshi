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
        self.book_history: Deque[Tuple[datetime, Optional[int], Optional[int], Optional[int]]] = deque(maxlen=500)

    def update_trade(self, price: int, size: int, side: Optional[str], ts: Optional[datetime] = None) -> None:
        if ts is None:
            ts = datetime.now(timezone.utc)
        self.trades.append(Trade(ts=ts, price=price, size=size, side=side))

    def update_mid(self, mid: float) -> None:
        self.mid_history.append((datetime.now(timezone.utc), mid))

    def update_book(self, yes_bid: Optional[int], yes_ask: Optional[int], spread: Optional[int]) -> None:
        self.book_history.append((datetime.now(timezone.utc), yes_bid, yes_ask, spread))

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

    def bid_momentum(self, window_sec: int = 30) -> float:
        now = datetime.now(timezone.utc)
        recent = [b for b in self.book_history if (now - b[0]).total_seconds() <= window_sec and b[1] is not None]
        if len(recent) < 2:
            return 0.0
        return float(recent[-1][1] - recent[0][1])

    def ask_momentum(self, window_sec: int = 30) -> float:
        now = datetime.now(timezone.utc)
        recent = [b for b in self.book_history if (now - b[0]).total_seconds() <= window_sec and b[2] is not None]
        if len(recent) < 2:
            return 0.0
        return float(recent[-1][2] - recent[0][2])

    def spread_trend(self, window_sec: int = 30) -> float:
        now = datetime.now(timezone.utc)
        recent = [b for b in self.book_history if (now - b[0]).total_seconds() <= window_sec and b[3] is not None]
        if len(recent) < 2:
            return 0.0
        return float(recent[-1][3] - recent[0][3])

    def realized_var(self, window_sec: int = 60) -> float:
        now = datetime.now(timezone.utc)
        series = [m[1] for m in self.mid_history if (now - m[0]).total_seconds() <= window_sec]
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        return sum((x - mean) ** 2 for x in series) / (len(series) - 1)
