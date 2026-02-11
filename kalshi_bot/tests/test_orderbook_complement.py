from kalshi_bot.market_selector import _orderbook_complement


def test_complement():
    ob = {"yes": [[40, 10]], "no": [[55, 5]]}
    prices = _orderbook_complement(ob)
    assert prices["best_yes_bid"] == 40
    assert prices["best_no_bid"] == 55
    assert prices["best_yes_ask"] == 45
    assert prices["best_no_ask"] == 60
