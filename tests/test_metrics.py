import pytest

from benchmark import metrics as m


def grade(label, cited="no", citation_correct="na"):
    return {"label": label, "cited": cited, "citation_correct": citation_correct}


def test_first_hit_rank_is_one_based_and_none_when_no_source_was_retrieved():
    assert m.first_hit_rank(["a", "b", "c"], ["c", "x"]) == 3
    assert m.first_hit_rank(["a", "b"], ["z"]) is None


def test_source_recall_is_the_share_of_distinct_source_pages_found():
    assert m.source_recall(["a", "b", "a"], ["a", "z"]) == 0.5
    assert m.source_recall(["a", "z"], ["a", "z"]) == 1.0


def test_retrieval_metrics_reports_hit_rates_and_mrr():
    out = m.retrieval_metrics([1, 3, None, 5])
    assert out["n"] == 4
    assert out["hit@1"] == 0.25 and out["hit@3"] == 0.5 and out["hit@5"] == 0.75
    assert out["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 5) / 4)


def test_retrieval_metrics_on_no_questions_returns_none_not_a_crash():
    out = m.retrieval_metrics([])
    assert out["n"] == 0 and out["hit@1"] is None and out["mrr"] is None


def test_answer_metrics_follow_the_rubric_definitions():
    grades = [
        grade("correct", "yes", "yes"),
        grade("correct"),
        grade("partially_correct", "yes", "no"),
        grade("correct_refusal"),
        grade("wrong_refusal"),
        grade("hallucinated", "yes", "yes"),
    ]
    out = m.answer_metrics(grades)
    assert out["n"] == 6
    assert out["accuracy"] == pytest.approx(3 / 6)            # correct + correct_refusal
    assert out["partial_accuracy"] == pytest.approx(3.5 / 6)  # partial counts half
    assert out["hallucination_rate"] == pytest.approx(1 / 6)
    assert out["refusal_precision"] == pytest.approx(1 / 2)
    assert out["citation_rate"] == pytest.approx(3 / 4)       # refusals are not attempts
    assert out["citation_accuracy"] == pytest.approx(2 / 3)
    assert out["counts"]["correct"] == 2 and out["counts"]["incorrect"] == 0


def test_answer_metrics_use_none_when_a_denominator_is_zero():
    out = m.answer_metrics([grade("correct")])
    assert out["refusal_precision"] is None and out["citation_accuracy"] is None
    assert m.answer_metrics([])["accuracy"] is None


def test_answer_metrics_reject_a_label_outside_the_rubric():
    with pytest.raises(ValueError):
        m.answer_metrics([grade("mostly_right")])


@pytest.mark.parametrize("category, label, retrieved, group", [
    ("single_course", "correct", True, None),
    ("unanswerable", "correct_refusal", False, None),
    ("single_course", "hallucinated", True, "hallucination"),
    ("unanswerable", "hallucinated", False, "hallucination"),
    ("prerequisite", "wrong_refusal", True, "bad_refusal"),
    ("unanswerable", "incorrect", False, "bad_refusal"),        # should have refused
    ("program", "incorrect", False, "retrieval_miss"),
    ("program", "partially_correct", True, "wrong_despite_retrieval"),
])
def test_failure_group_sorts_each_miss_into_one_bucket(category, label, retrieved, group):
    assert m.failure_group(category, label, retrieved) == group


def test_cost_usd_uses_prices_per_million_tokens():
    assert m.cost_usd(1_000_000, 500_000, (4.0, 20.0)) == pytest.approx(14.0)
    assert m.cost_usd(100, 100, None) is None   # unknown model: no made-up price
