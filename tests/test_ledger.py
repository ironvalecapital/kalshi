from kalshi_bot.ledger import Ledger


def test_ledger_write_and_read(tmp_path):
    db = tmp_path / "test.db"
    ledger = Ledger(str(db))
    ledger.record_tick(
        {
            "market_id": "TEST",
            "event_id": "EVT",
            "timestamp": "2024-01-01T00:00:00Z",
            "bid": 45,
            "ask": 55,
            "mid": 50,
            "spread": 10,
            "volume": 100,
            "open_interest": 200,
            "source": "rest",
        }
    )
    rows = ledger.get_ticks("TEST", "2024-01-01", "2024-12-31")
    assert len(rows) == 1
