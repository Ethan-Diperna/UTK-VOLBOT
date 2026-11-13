import csv
import io

import pytest

from benchmark import grade

SHEET_ROWS = [
    {"item": "q1-A", "id": "q1", "question": "hours for COSC 311?", "expected_answer": "3",
     "source_urls": "https://catalog.utk.edu/a", "response_label": "A", "response_text": "It is 3 hours."},
    {"item": "q1-B", "id": "q1", "question": "hours for COSC 311?", "expected_answer": "3",
     "source_urls": "https://catalog.utk.edu/a", "response_label": "B", "response_text": "It is 4 hours."},
    {"item": "q2-A", "id": "q2", "question": "who teaches it?", "expected_answer": "Not in the catalog.",
     "source_urls": "", "response_label": "A", "response_text": "The catalog does not say."},
]


def make_sheet(tmp_path):
    path = tmp_path / "blind_sheet.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SHEET_ROWS[0]))
        writer.writeheader()
        writer.writerows(SHEET_ROWS)
    return path


def session(tmp_path, typed, **kwargs):
    sheet, grades, out = make_sheet(tmp_path), tmp_path / "human_grades.csv", io.StringIO()
    done = grade.run(sheet, grades, stream=io.StringIO(typed), out=out, **kwargs)
    return done, grade.load_grades(grades), out.getvalue()


def test_number_keys_map_to_the_six_rubric_labels():
    assert list(grade.LABEL_KEYS.values()) == ["correct", "partially_correct", "incorrect",
                                               "correct_refusal", "wrong_refusal", "hallucinated"]


def test_grades_are_saved_with_label_notes_and_timestamp(tmp_path):
    done, grades, _ = session(tmp_path, "1\n\n6\nsays 4 hours, catalog says 3\n4\nfine\n")
    assert done == 3
    assert grades["q1-A"]["label"] == "correct" and grades["q1-A"]["notes"] == ""
    assert grades["q1-B"]["label"] == "hallucinated" and "4 hours" in grades["q1-B"]["notes"]
    assert grades["q2-A"]["label"] == "correct_refusal" and grades["q2-A"]["graded_at"].endswith("+00:00")


def test_screen_shows_question_expected_answer_and_response_but_no_system_name(tmp_path):
    _, _, shown = session(tmp_path, "q\n")
    assert "hours for COSC 311?" in shown and "Expected answer: 3" in shown
    assert "Response A" in shown and "It is 3 hours." in shown
    words = shown.lower().replace("\n", " ").split()
    assert "ours" not in words and "baseline" not in words and "ut verse" not in shown.lower()


def test_invalid_key_asks_again_and_hallucinated_requires_a_quote(tmp_path):
    done, grades, shown = session(tmp_path, "9\nyes\n6\n\nit says 4 hours\nq\n")
    assert done == 1 and grades["q1-A"]["label"] == "hallucinated"
    assert grades["q1-A"]["notes"] == "it says 4 hours"
    assert shown.count("Type 1-6") >= 2 and "quote the false claim" in shown.lower()


def test_skip_leaves_the_item_ungraded_and_quit_stops(tmp_path):
    done, grades, _ = session(tmp_path, "s\n2\nonly half\nq\n")
    assert done == 1 and "q1-A" not in grades and grades["q1-B"]["label"] == "partially_correct"


def test_second_session_resumes_with_only_the_ungraded_items(tmp_path):
    sheet, grades_path = make_sheet(tmp_path), tmp_path / "human_grades.csv"
    grade.run(sheet, grades_path, stream=io.StringIO("1\n\nq\n"), out=io.StringIO())
    out = io.StringIO()
    done = grade.run(sheet, grades_path, stream=io.StringIO("3\n\n4\n\n"), out=out)
    assert done == 2 and len(grade.load_grades(grades_path)) == 3
    assert "It is 3 hours." not in out.getvalue()        # q1-A was already graded


def test_redo_replaces_an_earlier_grade(tmp_path):
    sheet, grades_path = make_sheet(tmp_path), tmp_path / "human_grades.csv"
    grade.run(sheet, grades_path, stream=io.StringIO("1\n\nq\n"), out=io.StringIO())
    grade.run(sheet, grades_path, stream=io.StringIO("2\nmissed the lab\n"), out=io.StringIO(), redo="q1-A")
    assert grade.load_grades(grades_path)["q1-A"]["label"] == "partially_correct"


def test_redo_list_shows_only_the_listed_items_and_can_be_resumed(tmp_path):
    sheet, grades_path = make_sheet(tmp_path), tmp_path / "human_grades.csv"
    grade.run(sheet, grades_path, stream=io.StringIO("1\n\n1\n\n1\n\n"), out=io.StringIO())       # first pass: all "correct"
    listed = ["q1-B", "q2-A"]
    first = grade.run(sheet, grades_path, stream=io.StringIO("6\nsays 4, catalog says 3\nq\n"), out=io.StringIO(),
                      redo=listed, skip_regraded=True)
    out = io.StringIO()
    second = grade.run(sheet, grades_path, stream=io.StringIO("4\n\n"), out=out, redo=listed, skip_regraded=True)
    grades = grade.load_grades(grades_path)
    assert (first, second) == (1, 1)
    assert grades["q1-A"]["label"] == "correct"                  # not listed, untouched
    assert grades["q1-B"]["label"] == "hallucinated" and grades["q2-A"]["label"] == "correct_refusal"
    assert "It is 4 hours." not in out.getvalue()                # q1-B was not shown a second time
    assert grade.grade_counts(grades_path) == {"q1-A": 1, "q1-B": 2, "q2-A": 2}


def test_redo_refuses_an_item_that_is_not_on_the_sheet(tmp_path):
    sheet, grades_path = make_sheet(tmp_path), tmp_path / "human_grades.csv"
    with pytest.raises(ValueError) as e:
        grade.run(sheet, grades_path, stream=io.StringIO(""), out=io.StringIO(), redo=["q1-A", "q9-Z"])
    assert "q9-Z" in str(e.value)


def test_item_list_file_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "flagged.txt"
    path.write_text("# why these\n\nq1-B\n  q2-A  \n# end\n", encoding="utf-8")
    assert grade.read_item_list(path) == ["q1-B", "q2-A"]


def test_end_of_input_stops_cleanly_without_saving_a_half_entered_grade(tmp_path):
    done, grades, _ = session(tmp_path, "6\n")           # label typed, note never given
    assert done == 0 and grades == {}
