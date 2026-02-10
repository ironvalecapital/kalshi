from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional, Tuple

from .config import RiskLimits


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    open_orders: int = 0
    positions: Dict[str, float] = field(default_factory=dict)
    last_trade_time: Dict[str, float] = field(default_factory=dict)
    consecutive_losses: int = 0
    last_loss_time: Optional[float] = None
    pause_until: Optional[float] = None


class CircuitBreaker:
    def __init__(self) -> None:
        self.rate_limit_hits: int = 0
        self.forbidden_hits: int = 0

    def note_api_error(self, status_code: int, retry_after: Optional[int] = None) -> Optional[float]:
        now = time.time()
        if status_code == 429:
            self.rate_limit_hits += 1
            return now + (retry_after or min(60, 5 * self.rate_limit_hits))
        if status_code == 403:
            self.forbidden_hits += 1
            return now + (retry_after or min(300, 30 * self.forbidden_hits))
        return None


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.state = RiskState()
        self.circuit = CircuitBreaker()
        self.current_day = date.today()

    def refresh_day(self) -> None:
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.state.daily_pnl = 0.0
            self.state.consecutive_losses = 0

    def note_trade(self, market_id: str, pnl: float) -> None:
        self.refresh_day()
        self.state.daily_pnl += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_time = time.time()
        else:
            self.state.consecutive_losses = 0
        self.state.last_trade_time[market_id] = time.time()

    def note_open_orders(self, count: int) -> None:
        self.state.open_orders = count

    def note_position(self, market_id: str, notional: float) -> None:
        self.state.positions[market_id] = notional

    def should_pause(self) -> Tuple[bool, Optional[str]]:
        now = time.time()
        if self.state.pause_until and now < self.state.pause_until:
            return True, "paused_by_circuit_breaker"
        if self.state.daily_pnl <= -abs(self.limits.max_daily_loss_usd):
            return True, "max_daily_loss_reached"
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            if self.state.last_loss_time and now - self.state.last_loss_time < self.limits.cooldown_seconds_after_loss:
                return True, "cooldown_after_losses"
        return False, None

    def note_api_error(self, status_code: int, retry_after: Optional[int] = None) -> None:
        pause_until = self.circuit.note_api_error(status_code, retry_after)
        if pause_until:
            self.state.pause_until = pause_until

    def check_order(
        self,
        market_id: str,
        order_size: int,
        order_notional: float,
    ) -> Tuple[bool, str]:
        self.refresh_day()
        if self.state.daily_pnl <= -abs(self.limits.max_daily_loss_usd):
            return False, "daily_loss_exceeded"
        if order_size > self.limits.max_order_size_contracts:
            return False, "max_order_size_exceeded"
        if order_notional > self.limits.max_notional_usd:
            return False, "max_notional_exceeded"
        if self.state.open_orders >= self.limits.max_open_orders:
            return False, "max_open_orders_exceeded"
        position = self.state.positions.get(market_id, 0.0)
        if abs(position + order_notional) > self.limits.max_position_per_market_usd:
            return False, "max_position_per_market_exceeded"
        last_trade = self.state.last_trade_time.get(market_id)
        if last_trade and time.time() - last_trade < self.limits.max_trade_freq_per_market_seconds:
            return False, "trade_frequency_exceeded"
        return True, "ok"
