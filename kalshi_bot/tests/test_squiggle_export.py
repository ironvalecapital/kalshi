from kalshi_bot.squiggle_export import is_focus_sports_market, normalize_market_row, render_squiggle_program


def test_focus_filter_detects_ufc_and_euroleague():
    assert is_focus_sports_market({"ticker": "KXUFC-TEST"})
    assert is_focus_sports_market({"title": "EuroLeague Final Winner"})
    assert not is_focus_sports_market({"ticker": "KXCPI-TEST", "title": "US CPI"})


def test_render_program_contains_market_rows():
    row = normalize_market_row(
        {
            "ticker": "KXNBA-TEST",
            "event_ticker": "NBA",
            "title": "NBA winner",
            "yes_bid": 61,
            "no_bid": 37,
            "close_time": "2026-02-20T00:00:00Z",
        }
    )
    txt = render_squiggle_program([row])
    assert "markets = [" in txt
    assert "KXNBA-TEST" in txt
    assert "kellyFraction" in txt
