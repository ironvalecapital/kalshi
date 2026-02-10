from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Portfolio:
    positions: Dict[str, float] = field(default_factory=dict)

    def update_position(self, market_id: str, notional: float) -> None:
        self.positions[market_id] = notional

    def get_position(self, market_id: str) -> float:
        return self.positions.get(market_id, 0.0)
