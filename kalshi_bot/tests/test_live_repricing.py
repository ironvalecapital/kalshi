from kalshi_bot.models.live_repricing import LiveState, monte_carlo_win_probability, win_probability


def test_win_probability_monotonic_score():
    p1 = win_probability(score_diff=2, time_remaining_sec=600, possession=0, efficiency_edge=0.0, pace=98)
    p2 = win_probability(score_diff=8, time_remaining_sec=600, possession=0, efficiency_edge=0.0, pace=98)
    assert p2 > p1


def test_win_probability_time_decay_effect():
    # Same lead should be safer with less time remaining.
    p_early = win_probability(score_diff=4, time_remaining_sec=1800, possession=0, efficiency_edge=0.0, pace=98)
    p_late = win_probability(score_diff=4, time_remaining_sec=120, possession=0, efficiency_edge=0.0, pace=98)
    assert p_late > p_early


def test_monte_carlo_probability_bounds():
    state = LiveState(score_diff=3, time_remaining_sec=720, possession=1, efficiency_edge=0.05, pace=99)
    p = monte_carlo_win_probability(state, simulations=500, seed=1)
    assert 0.0 < p < 1.0


def test_monte_carlo_efficiency_edge_directional():
    base = LiveState(score_diff=0, time_remaining_sec=600, possession=0, efficiency_edge=0.0, pace=99)
    plus = LiveState(score_diff=0, time_remaining_sec=600, possession=0, efficiency_edge=0.08, pace=99)
    p_base = monte_carlo_win_probability(base, simulations=1000, seed=7)
    p_plus = monte_carlo_win_probability(plus, simulations=1000, seed=7)
    assert p_plus > p_base


def test_win_probability_extreme_inputs_do_not_overflow():
    p_low = win_probability(score_diff=-1000, time_remaining_sec=2880, possession=-1, efficiency_edge=-2.0, pace=120)
    p_high = win_probability(score_diff=1000, time_remaining_sec=1, possession=1, efficiency_edge=2.0, pace=120)
    assert 0.0 < p_low < 1.0
    assert 0.0 < p_high < 1.0
