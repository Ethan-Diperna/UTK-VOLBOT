import json

import pytest

from benchmark import judge_sheet, score
from benchmark.judge import JudgeError

A_URL = "https://catalog.utk.edu/preview_course_nopop.php?catoid=56&coid=1"
QUESTIONS = [
    {"id": "q1", "question": "hours?", "expected_answer": "3", "source_urls": [A_URL], "category": "single_course"},
    {"id": "q2", "question": "prereq?", "expected_answer": "COSC 202", "source_urls": [A_URL], "category": "prerequisite"},
    {"id": "q3", "question": "who teaches?", "expected_answer": "Not in the catalog.", "source_urls": [],
     "category": "unanswerable"},
]
KEY = {"q1": {"A": "ours", "B": "baseline"}, "q2": {"A": "baseline", "B": "ours"}, "q3": {"A": "ours", "B": "baseline"}}


def grades(labels):
    """{'q1-A': 'correct', ...} -> grade rows keyed by item."""
    return {item: {"item": item, "id": item.split("-")[0], "label": label} for item, label in labels.items()}


HUMAN = grades({"q1-A": "correct", "q1-B": "hallucinated", "q2-A": "wrong_refusal", "q2-B": "correct",
                "q3-A": "correct_refusal", "q3-B": "correct_refusal"})


def test_unblind_maps_each_grade_to_the_system_that_wrote_the_response():
    by_system = score.unblind(HUMAN, KEY)
    assert by_system["ours"] == {"q1": "correct", "q2": "correct", "q3": "correct_refusal"}
    assert by_system["baseline"] == {"q1": "hallucinated", "q2": "wrong_refusal", "q3": "correct_refusal"}


def test_unblind_refuses_when_a_grade_is_missing():
    incomplete = {k: v for k, v in HUMAN.items() if k != "q2-B"}
    with pytest.raises(ValueError) as e:
        score.unblind(incomplete, KEY)
    assert "q2-B" in str(e.value)


def test_system_summary_gives_accuracy_interval_rates_and_category_breakdown():
    out = score.system_summary(score.unblind(HUMAN, KEY)["baseline"], QUESTIONS)
    assert out["n"] == 3 and out["accuracy"] == pytest.approx(1 / 3)
    assert out["accuracy_ci"][0] <= 1 / 3 <= out["accuracy_ci"][1]
    assert out["hallucination_rate"] == pytest.approx(1 / 3)
    assert out["counts"]["wrong_refusal"] == 1
    assert out["by_category"]["unanswerable"] == {"n": 1, "accuracy": 1.0}
    assert out["by_category"]["single_course"] == {"n": 1, "accuracy": 0.0}
    assert "citation_rate" not in out               # citations are scored in their own pass


def test_compare_reports_the_gap_its_interval_and_mcnemar():
    by_system = score.unblind(HUMAN, KEY)
    out = score.compare(by_system["ours"], by_system["baseline"], QUESTIONS)
    assert out["accuracy_difference"] == pytest.approx(2 / 3)
    assert out["mcnemar"]["ours_only"] == 2 and out["mcnemar"]["baseline_only"] == 0
    assert out["mcnemar"]["p_value"] == 0.5         # 2 discordant pairs cannot be significant


def test_grader_agreement_reports_exact_and_right_wrong_agreement():
    judge = grades({"q1-A": "correct", "q1-B": "incorrect", "q2-A": "wrong_refusal", "q2-B": "partially_correct",
                    "q3-A": "correct_refusal", "q3-B": "correct"})
    out = score.grader_agreement(HUMAN, judge)
    assert out["n"] == 6
    assert out["exact_agreement"] == pytest.approx(3 / 6)
    assert out["right_wrong_agreement"] == pytest.approx(5 / 6)    # only q2-B flips right/wrong
    assert -1 <= out["kappa"] <= 1


@pytest.mark.parametrize("cited, expected", [
    ("COSC 360 - Systems Programming - - UTK Catalog", True),
    ("MATH 141* - Calculus I - - UTK Catalog", False),
    ("Program: Computer Science Major, BS in ... - UTK Catalog", False),
    ("Course Descriptions - University of Tennessee, Knoxville", False),
])
def test_titles_match_across_catalog_years_and_site_suffixes(cited, expected):
    assert score.titles_match(cited, "COSC 360 - Systems Programming") is expected


def test_truncated_program_title_matches_its_full_title():
    assert score.titles_match("Program: Applied Artificial Intelligence Major, BS in ... - UTK Catalog",
                              "Applied Artificial Intelligence Major, BS in Applied Artificial Intelligence")


def test_citation_fields_follow_the_rubric():
    titles = {A_URL: "COSC 360 - Systems Programming"}
    q = QUESTIONS[0]
    by_url = score.citation_fields(q, [{"title": "x", "url": A_URL}], titles)
    by_title = score.citation_fields(q, [{"title": "COSC 360 - Systems Programming - - UTK Catalog",
                                          "url": "https://catalog.utk.edu/preview_course_nopop.php?catoid=20&coid=9"}], titles)
    wrong = score.citation_fields(q, [{"title": "Reddit", "url": "https://www.reddit.com/r/UTK"}], titles)
    none = score.citation_fields(q, [], titles)
    assert by_url == {"cited": "yes", "citation_correct": "yes"}
    assert by_title == {"cited": "yes", "citation_correct": "yes"}
    assert wrong == {"cited": "yes", "citation_correct": "no"}
    assert none == {"cited": "no", "citation_correct": "na"}


def test_citation_summary_covers_answerable_attempts_with_known_sources_only():
    labels = {"q1": "correct", "q2": "wrong_refusal", "q3": "correct_refusal"}
    cites = {"q1": [{"title": "t", "url": A_URL}], "q3": [{"title": "t", "url": A_URL}]}   # q2 unknown
    out = score.citation_summary(labels, cites, QUESTIONS, {A_URL: "COSC 360 - Systems Programming"})
    assert out == {"n_scored": 1, "n_without_citation_data": 1, "citation_rate": 1.0, "citation_accuracy": 1.0}


def test_key_is_only_accepted_when_it_matches_the_committed_hash(tmp_path):
    key_path, hash_path = tmp_path / "blind_key.json", tmp_path / "blind_key.sha256"
    key_path.write_text(json.dumps({"seed": 1, "assignments": KEY}), encoding="utf-8")
    import hashlib
    hash_path.write_text(hashlib.sha256(key_path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    assert score.load_key(key_path, hash_path) == KEY
    key_path.write_text(json.dumps({"seed": 1, "assignments": {**KEY, "q1": {"A": "baseline", "B": "ours"}}}),
                        encoding="utf-8")
    with pytest.raises(SystemExit):
        score.load_key(key_path, hash_path)


SHEET = [{"item": f"{q['id']}-{label}", "id": q["id"], "response_label": label, "response_text": f"text {q['id']} {label}"}
         for q in QUESTIONS for label in ("A", "B")]


def test_judge_sheet_grades_only_items_not_done_yet_and_never_sees_the_key():
    seen = []

    def judge_fn(question, response_text):
        seen.append((question["id"], response_text))
        return {"label": "correct", "notes": "ok", "judge_model": "claude-sonnet-5-5"}

    rows, errors = judge_sheet.judge_items(SHEET, QUESTIONS, judge_fn, done={"q1-A", "q1-B"}, workers=1)
    assert [r["item"] for r in rows] == ["q2-A", "q2-B", "q3-A", "q3-B"] and errors == []
    assert rows[0]["label"] == "correct" and rows[0]["judge_model"] == "claude-sonnet-5-5"
    assert ("q2", "text q2 A") in seen and len(seen) == 4


def test_judge_sheet_reports_failures_and_keeps_going():
    def judge_fn(question, response_text):
        if question["id"] == "q2":
            raise JudgeError("cut off")
        return {"label": "correct", "notes": "", "judge_model": "m"}

    rows, errors = judge_sheet.judge_items(SHEET, QUESTIONS, judge_fn, done=set(), workers=1)
    assert len(rows) == 4 and sorted(e[0] for e in errors) == ["q2-A", "q2-B"]
