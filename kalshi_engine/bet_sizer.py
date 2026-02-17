from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetSizingResult:
    kelly_fraction: float
    fractional_kelly: float
    dollars: float
    contracts: int


def kelly_fraction_binary(p_yes: float, price_prob: float) -> float:
    """
    Kelly fraction for binary contract.
    price_prob is contract price in [0,1] for YES.
    """
    p = max(1e-9, min(1.0 - 1e-9, float(p_yes)))
    q = 1.0 - p
    c = max(1e-9, min(1.0 - 1e-9, float(price_prob)))
    b = (1.0 - c) / c
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))


def suggest_contracts(
    bankroll_usd: float,
    price_cents: int,
    p_yes: float,
    fractional_kelly: float = 0.25,
    max_risk_fraction: float = 0.03,
) -> BetSizingResult:
    if bankroll_usd <= 0 or price_cents <= 0 or price_cents >= 100:
        return BetSizingResult(0.0, 0.0, 0.0, 0)
    price_prob = price_cents / 100.0
    k = kelly_fraction_binary(p_yes, price_prob)
    frac = max(0.0, min(1.0, float(fractional_kelly)))
    f = min(k * frac, max(0.0, float(max_risk_fraction)))
    dollars = bankroll_usd * f
    contracts = int(dollars / price_prob)
    return BetSizingResult(kelly_fraction=k, fractional_kelly=f, dollars=dollars, contracts=max(0, contracts))
