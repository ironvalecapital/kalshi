from kalshi_bot.models.lmsr_bayes import bayesian_yes_probability, lmsr_yes_probability


def test_lmsr_probability_moves_with_depth_imbalance():
    p_yes_heavy = lmsr_yes_probability(depth_yes=300, depth_no=100, b=30.0)
    p_no_heavy = lmsr_yes_probability(depth_yes=100, depth_no=300, b=30.0)
    assert p_yes_heavy > 0.5
    assert p_no_heavy < 0.5


def test_bayesian_probability_blends_prior_and_evidence():
    out = bayesian_yes_probability(
        prior_yes=0.5,
        prior_weight=2.0,
        evidences=[(0.7, 3.0), (0.6, 2.0)],
    )
    assert 0.5 < out.posterior_yes < 0.7


def test_bayesian_probability_clamps_extremes():
    out = bayesian_yes_probability(
        prior_yes=0.9999,
        prior_weight=5.0,
        evidences=[(1.0, 100.0)],
    )
    assert 0.0 < out.posterior_yes < 1.0
