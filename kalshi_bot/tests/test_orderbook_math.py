from kalshi_bot.market_picker import orderbook_best_prices


def test_orderbook_ask_derivation():
    ob = {"yes": [[45, 10]], "no": [[52, 7]]}
    prices = orderbook_best_prices(ob)
    assert prices["best_yes_bid"] == 45
    assert prices["best_no_bid"] == 52
    assert prices["best_yes_ask"] == 48
    assert prices["best_no_ask"] == 55
    assert prices["spread_yes"] == 3
