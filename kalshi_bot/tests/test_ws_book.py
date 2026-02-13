from kalshi_bot.orderbook_live import OrderbookState, LiveOrderbook


def test_orderbook_snapshot_delta():
    state = OrderbookState(yes_bids={}, no_bids={})
    ob = LiveOrderbook.__new__(LiveOrderbook)
    ob.state = state

    snapshot = {"msg": {"yes": [[45, 10]], "no": [[55, 12]]}}
    ob.apply_snapshot(snapshot)
    assert ob.state.best_yes_bid() == 45
    assert ob.state.best_no_bid() == 55
    assert ob.state.best_yes_ask() == 45  # 100-55

    delta = {"msg": {"side": "yes", "price": 46, "size": 7}}
    ob.apply_delta(delta)
    assert ob.state.best_yes_bid() == 46
