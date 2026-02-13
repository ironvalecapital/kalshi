from kalshi_bot.fee_model import fee_cents, fee_total_cents, fee_per_contract_cents


def test_fee_rounding():
    assert fee_cents(1, 50, maker=True) >= 0
    assert fee_cents(1, 1, maker=True) >= 0
    assert fee_cents(1, 99, maker=False) >= 0
    assert fee_cents(10, 50, maker=False) >= fee_cents(10, 50, maker=True)


def test_fee_total_and_per_contract():
    total = fee_total_cents(10, 50, maker=True)
    per = fee_per_contract_cents(10, 50, maker=True)
    assert total >= per


def test_fee_maker_taker_gap():
    maker = fee_cents(10, 50, maker=True)
    taker = fee_cents(10, 50, maker=False)
    assert taker >= maker
