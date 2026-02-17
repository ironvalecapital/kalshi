from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple
import random


@dataclass
class RLState:
    market_type: str
    volatility_regime: str
    liquidity_score: float
    emotion_score: float
    time_to_expiry_min: float
    edge_size: float
    spread_width: float
    depth_imbalance: float
    drawdown_level: float


@dataclass(frozen=True)
class BanditAction:
    mode: str  # GTO / EXPLOIT
    kelly_multiplier: float  # 0.15 / 0.25 / 0.35
    edge_threshold: float  # 0.04 / 0.06 / 0.08
    execution_style: str  # passive / aggressive


def build_action_space() -> List[BanditAction]:
    return [
        BanditAction(mode=m, kelly_multiplier=k, edge_threshold=e, execution_style=x)
        for m, k, e, x in product(
            ["GTO", "EXPLOIT"],
            [0.15, 0.25, 0.35],
            [0.04, 0.06, 0.08],
            ["passive", "aggressive"],
        )
    ]


@dataclass
class ExecutionControls:
    mode: str
    edge_threshold: float
    kelly_fraction: float
    execution_style: str


def risk_adjusted_reward(
    realized_pnl: float,
    variance_penalty: float,
    drawdown_penalty: float,
    lambda_var: float = 0.7,
    mu_dd: float = 1.0,
) -> float:
    return float(realized_pnl) - float(lambda_var) * float(variance_penalty) - float(mu_dd) * float(drawdown_penalty)


def ev_ratio_reward(realized_pnl: float, expected_pnl: float) -> float:
    if abs(expected_pnl) < 1e-9:
        return 0.0
    return float(realized_pnl) / float(expected_pnl)


class ContextualBanditOptimizer:
    """
    Epsilon-greedy contextual bandit over discretized state buckets.
    Tracks mean reward per (state, action) pair.
    """

    def __init__(
        self,
        epsilon: float = 0.10,
        min_learning_rate: float = 0.02,
        actions: List[BanditAction] | None = None,
        seed: int = 7,
    ) -> None:
        self.epsilon = float(epsilon)
        self.min_learning_rate = float(min_learning_rate)
        self.actions = actions or build_action_space()
        self.rng = random.Random(seed)
        self.mean_reward: Dict[Tuple[str, int], float] = {}
        self.counts: Dict[Tuple[str, int], int] = {}

    def _bucket(self, v: float, cuts: List[float]) -> int:
        for i, c in enumerate(cuts):
            if v < c:
                return i
        return len(cuts)

    def state_key(self, s: RLState) -> str:
        liq_b = self._bucket(s.liquidity_score, [0.3, 0.6])
        emo_b = self._bucket(s.emotion_score, [0.8, 1.5])
        tte_b = self._bucket(s.time_to_expiry_min, [30, 180])
        edge_b = self._bucket(s.edge_size, [0.02, 0.06])
        spr_b = self._bucket(s.spread_width, [2, 6])
        imb_b = self._bucket(abs(s.depth_imbalance), [0.25, 0.55])
        dd_b = self._bucket(s.drawdown_level, [0.10, 0.20])
        return f"{s.market_type}:{s.volatility_regime}:{liq_b}:{emo_b}:{tte_b}:{edge_b}:{spr_b}:{imb_b}:{dd_b}"

    def select_action_idx(self, state: RLState, explore: bool = True) -> int:
        key = self.state_key(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.actions))
        vals = [self.mean_reward.get((key, i), 0.0) for i in range(len(self.actions))]
        return max(range(len(vals)), key=lambda i: vals[i])

    def select_action(self, state: RLState, explore: bool = True) -> BanditAction:
        return self.actions[self.select_action_idx(state, explore=explore)]

    def update(self, state: RLState, action_idx: int, reward: float) -> None:
        key = (self.state_key(state), int(action_idx))
        n = self.counts.get(key, 0) + 1
        old = self.mean_reward.get(key, 0.0)
        lr = max(self.min_learning_rate, 1.0 / n)
        self.mean_reward[key] = old + lr * (float(reward) - old)
        self.counts[key] = n


def controls_from_action(action: BanditAction, drawdown_level: float) -> ExecutionControls:
    edge = float(action.edge_threshold)
    kelly = float(action.kelly_multiplier)

    # Drawdown guardrails for live stability.
    if drawdown_level >= 0.15:
        edge *= 1.10
    if drawdown_level >= 0.20:
        kelly *= 0.50

    return ExecutionControls(
        mode=action.mode,
        edge_threshold=max(0.02, min(0.12, edge)),
        kelly_fraction=max(0.05, min(0.35, kelly)),
        execution_style=action.execution_style,
    )
