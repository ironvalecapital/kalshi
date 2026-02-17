from .bet_sizer import BetSizingResult, kelly_fraction_binary, suggest_contracts
from .monte_carlo_bayes import BetaPosterior, bayesian_probability_from_stream, beta_update, monte_carlo_probability
from .pipeline import MarketDecision, build_market_decision, decision_to_dict

try:
    from .model_binary import train_binary_model
except Exception:  # pragma: no cover - optional runtime dependency (lightgbm/libomp)
    train_binary_model = None  # type: ignore[assignment]

__all__ = [
    "BetSizingResult",
    "BetaPosterior",
    "MarketDecision",
    "bayesian_probability_from_stream",
    "beta_update",
    "build_market_decision",
    "decision_to_dict",
    "kelly_fraction_binary",
    "monte_carlo_probability",
    "suggest_contracts",
    "train_binary_model",
]
