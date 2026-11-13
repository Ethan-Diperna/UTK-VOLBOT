"""Statistics for the head-to-head. Pure functions, standard library only.

Three questions, three tools:

* How sure is one accuracy number?  -> bootstrap confidence interval.
* Is the gap between two systems real?  -> McNemar's exact test, because both
  systems answered the same questions (paired data).
* Do two graders agree?  -> raw agreement and Cohen's kappa.
"""
from __future__ import annotations

import math
import random
from collections import Counter

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261113
CONFIDENCE_LEVEL = 0.95


def _percentile_interval(samples: list[float], level: float) -> tuple[float, float]:
    ordered = sorted(samples)
    tail = (1 - level) / 2
    low = ordered[int(math.floor(tail * len(ordered)))]
    high = ordered[int(math.ceil((1 - tail) * len(ordered))) - 1]
    return low, high


def bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                 level: float = CONFIDENCE_LEVEL) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for the mean of `values` (1 = right,
    0 = wrong gives an interval for accuracy).

    Idea: the test questions are a sample of all questions students could
    ask. Draw a new set of the same size from them with replacement, compute
    the accuracy, repeat 10,000 times, and take the middle 95% of those
    accuracies. No formula or normality assumption is needed."""
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_resamples)]
    return _percentile_interval(means, level)


def paired_bootstrap_diff(ours: list[float], baseline: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                          seed: int = BOOTSTRAP_SEED, level: float = CONFIDENCE_LEVEL
                          ) -> tuple[float, float, float]:
    """(difference, low, high) for mean(ours) - mean(baseline).

    Questions are resampled, and each resampled question brings both systems'
    results with it. That keeps the pairing: a hard question is hard for both,
    and resampling the two systems separately would throw that away."""
    if len(ours) != len(baseline):
        raise ValueError(f"paired lists differ in length: {len(ours)} vs {len(baseline)}")
    diffs = [a - b for a, b in zip(ours, baseline)]
    n = len(diffs)
    rng = random.Random(seed)
    means = [sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples)]
    low, high = _percentile_interval(means, level)
    return sum(diffs) / n, low, high


def mcnemar_exact(ours_right: list[bool], baseline_right: list[bool]) -> dict:
    """McNemar's exact test on paired right/wrong outcomes.

    Only the questions where the systems disagree carry information. If the
    systems were equally good, each such question would be a coin flip for
    which one got it right. The p-value is the chance of a split at least as
    lopsided as the observed one under that coin-flip assumption (two-sided
    binomial test with p = 0.5)."""
    if len(ours_right) != len(baseline_right):
        raise ValueError("paired lists differ in length")
    pairs = Counter(zip(ours_right, baseline_right))
    ours_only, baseline_only = pairs[(True, False)], pairs[(False, True)]
    n = ours_only + baseline_only
    if n == 0:
        p_value = 1.0
    else:
        smaller = min(ours_only, baseline_only)
        p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(smaller + 1)) / 2 ** n)
    return {"ours_only": ours_only, "baseline_only": baseline_only, "both": pairs[(True, True)],
            "neither": pairs[(False, False)], "p_value": p_value}


def agreement(a: list[str], b: list[str]) -> float | None:
    """Share of items two graders labelled identically."""
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else None


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement corrected for chance: (observed - expected) / (1 - expected),
    where expected is how often the two graders would agree if each just used
    their own label frequencies at random. 1 is perfect, 0 is chance level.
    None when chance agreement is already 1 (both used a single label)."""
    observed = agreement(a, b)
    if observed is None:
        return None
    n = len(a)
    freq_a, freq_b = Counter(a), Counter(b)
    expected = sum(freq_a[label] * freq_b[label] for label in freq_a) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)
