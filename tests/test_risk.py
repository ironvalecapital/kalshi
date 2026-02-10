from kalshi_bot.config import RiskLimits
from kalshi_bot.risk import RiskManager


def test_risk_limits_order_size():
    limits = RiskLimits(max_order_size_contracts=2)
    risk = RiskManager(limits)
    ok, reason = risk.check_order("ABC", 3, 1.0)
    assert not ok
    assert reason == "max_order_size_exceeded"


def test_risk_daily_loss():
    limits = RiskLimits(max_daily_loss_usd=10.0)
    risk = RiskManager(limits)
    risk.note_trade("ABC", -20.0)
    ok, reason = risk.check_order("ABC", 1, 1.0)
    assert not ok
    assert reason == "daily_loss_exceeded"
