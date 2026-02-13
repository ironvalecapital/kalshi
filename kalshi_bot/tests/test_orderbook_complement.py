from kalshi_bot.market_selector import _orderbook_complement


def test_complement():
    ob = {"yes": [[40, 10]], "no": [[55, 5]]}
    prices = _orderbook_complement(ob)
    assert prices["best_yes_bid"] == 40
    assert prices["best_no_bid"] == 55
    assert prices["best_yes_ask"] == 45
    assert prices["best_no_ask"] == 60


def test_orderbook_state_complement():
    from kalshi_bot.orderbook_live import OrderbookState

    state = OrderbookState(yes_bids={60: 1}, no_bids={20: 1})
    assert state.best_yes_ask() == 80
    assert state.best_no_ask() == 40
