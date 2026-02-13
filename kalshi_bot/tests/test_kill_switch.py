import os

from kalshi_bot.execution import ExecutionEngine, OrderRequest
from kalshi_bot.risk import RiskManager
from kalshi_bot.config import ExecutionConfig, RiskLimits


class DummyClient:
    def create_order(self, payload):
        return {"status": "ok"}

    def cancel_order(self, order_id):
        return {"status": "ok"}


class DummyLedger:
    def record_order(self, *args, **kwargs):
        pass

    def record_audit(self, *args, **kwargs):
        pass


def test_kill_switch_blocks_order(monkeypatch):
    os.environ["KALSHI_BOT_KILL"] = "1"
    engine = ExecutionEngine(DummyClient(), DummyLedger(), RiskManager(RiskLimits()), ExecutionConfig())
    try:
        engine.place_order(
            OrderRequest(market_id="X", side="yes", action="buy", price_cents=10, count=1, client_order_id="t")
        )
        assert False, "Expected kill switch to block"
    except RuntimeError:
        assert True
    finally:
        os.environ.pop("KALSHI_BOT_KILL", None)
