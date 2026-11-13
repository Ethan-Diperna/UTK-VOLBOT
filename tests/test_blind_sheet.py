import csv
import json

import pytest

from benchmark import make_blind_sheet as mbs

QUESTIONS = [
    {"id": f"q{i}", "question": f"question {i}?", "expected_answer": f"answer {i}",
     "source_urls": ["https://catalog.utk.edu/a", "https://catalog.utk.edu/b"], "category": "single_course"}
    for i in range(1, 41)
]
OURS = {q["id"]: f"ours says {q['id']} [1]." for q in QUESTIONS}
BASELINE = {q["id"]: f"baseline says {q['id']}.\n1\n\n\nSources\n3\n1 Referenced\nLearn More" for q in QUESTIONS}


def test_clean_removes_inline_citation_markers_and_the_space_they_leave():
    assert mbs.clean_response("COSC 360 is 4 credit hours [1]. It has a lab [2][3], too [1, 4].") == \
        "COSC 360 is 4 credit hours. It has a lab, too."


def test_clean_removes_copy_button_citation_lines_and_rejoins_the_sentence():
    raw = ("The catalog lists MATH 251 as Linear Algebra I.\n1\n Some older pages differ.\n3\n4\n\n"
           "Check with your advisor.\n\n\nSources\n9\n4 Referenced\nLearn More\n6 Images")
    assert mbs.clean_response(raw) == ("The catalog lists MATH 251 as Linear Algebra I. Some older pages differ.\n\n"
                                       "Check with your advisor.")


@pytest.mark.parametrize("trailer", [
    "Sources\n10\nClick to view sources\n3 Referenced\nLearn More",
    "Sources\n9\nClick to view sources\n3 Referenced\nLearn More\n1 Images",
    "Sources\n10\nClick to view sources",
    "Sources\n2\n1 Referenced\nLearn More\n6 Images",
])
def test_clean_removes_every_shape_of_the_sources_bar(trailer):
    assert mbs.clean_response(f"Check the syllabus for the title.\n1\n\n\n{trailer}") == "Check the syllabus for the title."


def test_clean_removes_the_table_toolbar_the_copy_button_drags_along():
    raw = "MATH 141\nCalculus I\nPlacement level 4\n\nCopy\n\nTranslate\n\nTTS\n\nExport\nBoth need trigonometry."
    assert mbs.clean_response(raw) == "MATH 141\nCalculus I\nPlacement level 4\nBoth need trigonometry."


def test_clean_collapses_a_line_repeated_right_after_itself():
    raw = "It is required.\n\nTypical degree roadmap\nTypical degree roadmap\nCheck your audit."
    assert mbs.clean_response(raw) == "It is required.\n\nTypical degree roadmap\nCheck your audit."


def test_clean_keeps_the_word_copy_in_a_normal_sentence():
    text = "Copy the form and send it to your advisor.\nExport options are listed there."
    assert mbs.clean_response(text) == text


def test_clean_keeps_the_word_sources_when_it_is_part_of_the_answer():
    text = "The catalog and the timetable are separate.\nSources of truth differ by office."
    assert mbs.clean_response(text) == text


def test_clean_removes_pasted_link_markup_citations():
    raw = ("It is likely ECE 202.\n\nI can't verify that.\n[1](https://utverse.tennessee.edu/#)\n"
           "[9](https://utverse.tennessee.edu/#)\n Check your degree audit.")
    assert mbs.clean_response(raw) == "It is likely ECE 202.\n\nI can't verify that. Check your degree audit."


def test_clean_strips_markdown_emphasis_headings_and_curly_quotes():
    raw = "## Prerequisites\n**COSC 302** or `COSC 307`, with a “C” or better. It’s required."
    assert mbs.clean_response(raw) == 'Prerequisites\nCOSC 302 or COSC 307, with a "C" or better. It\'s required.'


def test_clean_keeps_numbers_that_are_content_and_list_items():
    raw = "It is 3 credit hours.\n- MATH 141 (4 credit hours)\n- MATH 142\nTerm 4 lists 15 hours."
    assert mbs.clean_response(raw) == raw


def test_clean_leaves_no_trace_of_which_system_wrote_it():
    for text in (mbs.clean_response(OURS["q1"]), mbs.clean_response(BASELINE["q1"])):
        assert "[" not in text and "Sources" not in text and "Referenced" not in text
        assert not any(line.strip().isdigit() for line in text.splitlines())


def test_sheet_has_both_responses_per_question_labelled_a_and_b_only():
    rows, key = mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=7)
    assert len(rows) == 80
    assert [r["response_label"] for r in rows[:4]] == ["A", "B", "A", "B"]
    assert rows[0]["item"] == "q1-A" and rows[1]["item"] == "q1-B"
    assert rows[0]["source_urls"] == "https://catalog.utk.edu/a | https://catalog.utk.edu/b"
    assert set(rows[0]) == {"item", "id", "question", "expected_answer", "source_urls",
                            "response_label", "response_text"}       # no system, no category
    for q in QUESTIONS:
        assert set(key[q["id"]].values()) == {"ours", "baseline"}


def test_key_tells_which_text_is_whose_and_assignment_is_actually_mixed():
    rows, key = mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=7)
    by_item = {r["item"]: r for r in rows}
    for q in QUESTIONS:
        ours_label = next(label for label, system in key[q["id"]].items() if system == "ours")
        assert by_item[f"{q['id']}-{ours_label}"]["response_text"].startswith("ours says")
    ours_as_a = sum(1 for q in QUESTIONS if key[q["id"]]["A"] == "ours")
    assert 8 <= ours_as_a <= 32                                      # shuffled, not constant


def test_same_seed_gives_same_sheet_and_other_seed_differs():
    assert mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=7) == mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=7)
    assert mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=7)[1] != mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=8)[1]


def test_build_refuses_when_a_system_has_no_answer_for_a_question():
    missing = {k: v for k, v in BASELINE.items() if k != "q3"}
    with pytest.raises(ValueError) as e:
        mbs.build_sheet(QUESTIONS, OURS, missing, seed=7)
    assert "q3" in str(e.value)


def test_write_saves_csv_and_key_and_never_overwrites_a_sheet(tmp_path):
    rows, key = mbs.build_sheet(QUESTIONS, OURS, BASELINE, seed=7)
    sheet, key_path = tmp_path / "blind_sheet.csv", tmp_path / "blind_key.json"
    mbs.write_sheet(rows, key, sheet, key_path, meta={"seed": 7})
    with sheet.open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 80 and saved[0]["item"] == "q1-A"
    assert "ours" not in sheet.read_text(encoding="utf-8").replace("ours says", "")
    assert json.loads(key_path.read_text(encoding="utf-8"))["assignments"] == key
    with pytest.raises(FileExistsError):
        mbs.write_sheet(rows, key, sheet, key_path, meta={"seed": 7})
