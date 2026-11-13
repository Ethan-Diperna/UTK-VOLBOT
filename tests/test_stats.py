import pytest

from benchmark import stats


def test_bootstrap_ci_brackets_the_mean_and_is_repeatable():
    values = [1] * 34 + [0] * 34
    low, high = stats.bootstrap_ci(values, seed=1)
    assert low < 0.5 < high
    assert 0.33 < low < 0.43 and 0.57 < high < 0.67      # about +/- 0.12 for n = 68
    assert stats.bootstrap_ci(values, seed=1) == (low, high)


def test_bootstrap_ci_of_a_constant_is_that_constant_and_empty_is_none():
    assert stats.bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0)
    assert stats.bootstrap_ci([]) == (None, None)


def test_paired_bootstrap_resamples_questions_not_systems():
    ours = [1] * 50 + [0] * 18
    baseline = [1] * 30 + [0] * 38
    diff, low, high = stats.paired_bootstrap_diff(ours, baseline, seed=1)
    assert diff == pytest.approx(20 / 68)
    assert 0 < low < diff < high                         # ours is better on every resample here
    same = stats.paired_bootstrap_diff(ours, ours, seed=1)
    assert same == (0.0, 0.0, 0.0)                       # identical systems: no spread at all


def test_paired_bootstrap_rejects_lists_of_different_length():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1, 0], [1])


def test_mcnemar_counts_discordant_pairs_and_gives_the_exact_p_value():
    ours = [True] * 10 + [False] * 2 + [True] * 5 + [False] * 3
    base = [False] * 10 + [True] * 2 + [True] * 5 + [False] * 3
    out = stats.mcnemar_exact(ours, base)
    assert (out["ours_only"], out["baseline_only"], out["both"], out["neither"]) == (10, 2, 5, 3)
    assert out["p_value"] == pytest.approx(158 / 4096)   # 2 * (C(12,0) + C(12,1) + C(12,2)) / 2**12


def test_mcnemar_with_no_disagreements_is_p_one_and_is_symmetric():
    assert stats.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    a, b = [True] * 7 + [False] * 1, [False] * 7 + [True] * 1
    assert stats.mcnemar_exact(a, b)["p_value"] == pytest.approx(stats.mcnemar_exact(b, a)["p_value"])
    assert stats.mcnemar_exact([True] * 3 + [False] * 3, [False] * 3 + [True] * 3)["p_value"] == 1.0


def test_agreement_and_cohens_kappa():
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "y", "y"]
    assert stats.agreement(a, b) == 0.75
    assert stats.cohen_kappa(a, b) == pytest.approx(0.5)     # observed 0.75, chance 0.5
    assert stats.cohen_kappa(a, a) == 1.0
    assert stats.cohen_kappa(["x", "x"], ["x", "x"]) is None  # chance agreement is 1: undefined
    assert stats.agreement([], []) is None
