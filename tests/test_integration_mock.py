from kalshi_bot.config import ExecutionConfig, RiskLimits
from kalshi_bot.execution import ExecutionEngine, OrderRequest
from kalshi_bot.ledger import Ledger
from kalshi_bot.risk import RiskManager


class DummyClient:
    def __init__(self):
        self.last_payload = None

    def create_order(self, payload):
        self.last_payload = payload
        return {"order": {"order_id": "ORD123", "client_order_id": payload.get("client_order_id"), "status": "open"}}

    def cancel_order(self, order_id):
        return {"order_id": order_id, "status": "canceled"}


def test_execution_place_order(tmp_path):
    db = tmp_path / "test.db"
    ledger = Ledger(str(db))
    risk = RiskManager(RiskLimits(max_order_size_contracts=5))
    exec_engine = ExecutionEngine(DummyClient(), ledger, risk, ExecutionConfig())
    order = OrderRequest(
        market_id="TEST",
        side="yes",
        action="buy",
        price_cents=55,
        count=1,
        client_order_id="cid",
    )
    result = exec_engine.place_order(order)
    assert result["status"] == "submitted"
