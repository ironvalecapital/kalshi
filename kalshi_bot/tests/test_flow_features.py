from kalshi_bot.flow_features import FlowFeatures


def test_features():
    f = FlowFeatures()
    f.update_mid(50)
    f.update_mid(51)
    assert f.momentum(60) != 0
    assert -1 <= f.imbalance(10, 5) <= 1
