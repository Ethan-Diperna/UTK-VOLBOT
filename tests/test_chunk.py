import pytest

from src import chunk

COURSE = {
    "id": "course:1", "url": "https://catalog.utk.edu/preview_course_nopop.php?catoid=56&coid=1",
    "title": "COSC 302 - Fake Course", "doc_type": "course", "dept": "COSC",
    "text": "COSC 302 - Fake Course\nCredit Hours: 4\nPrerequisite(s): COSC 202 with a grade of C or better.",
    "metadata": {"code": "COSC 302", "catalog_year": "2026-2027"},
}

PROGRAM_TEXT = "\n".join([
    "Fake Major, BS",
    "",
    "Intro paragraph about the fake major with enough words to stand alone as an overview chunk here.",
    "",
    "## Campus: Knoxville",
    "",
    "## Term 1",
    "COSC 103 - Introduction to Computer Science 3 Credit Hours",
    "MATH 141* - Calculus I 4 Credit Hours",
    "ENGL 101* - English Composition I 3 Credit Hours",
    "",
    "### Term 1 Milestone Notes",
    "MATH 130 - Precalculus I 4 Credit Hours or a higher math course",
    "",
    "## Term 2",
    "COSC 203 - Data Structures 3 Credit Hours",
    "COSC 204 - Data Structures Laboratory 1 Credit Hours",
    "MATH 142* - Calculus II 4 Credit Hours",
    "",
    "## Footnotes",
    "Students must earn a C or better in every COSC course used for the degree requirements listed.",
])

PROGRAM = {
    "id": "program:9", "url": "https://catalog.utk.edu/preview_program.php?catoid=56&poid=9",
    "title": "Fake Major, BS", "doc_type": "program", "dept": "",
    "text": PROGRAM_TEXT, "metadata": {"catalog_year": "2026-2027"},
}


def body_lines(text):
    return [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("#")]


def test_course_is_exactly_one_chunk_with_text_and_code_unchanged():
    chunks = chunk.chunk_docs([COURSE])
    assert len(chunks) == 1
    c = chunks[0]
    assert c["text"] == COURSE["text"]
    assert "COSC 302" in c["text"]
    assert (c["doc_id"], c["url"], c["title"]) == (COURSE["id"], COURSE["url"], COURSE["title"])
    assert c["code"] == "COSC 302"


def test_split_blocks_reads_levels_headings_and_overview():
    blocks = chunk.split_blocks(PROGRAM_TEXT)
    assert [(b["level"], b["heading"]) for b in blocks] == [
        (1, "Overview"), (2, "Campus: Knoxville"), (2, "Term 1"),
        (3, "Term 1 Milestone Notes"), (2, "Term 2"), (2, "Footnotes")]
    assert blocks[2]["path"] == ["Term 1"]
    assert blocks[3]["path"] == ["Term 1", "Term 1 Milestone Notes"]


def test_split_blocks_treats_empty_heading_as_continuation():
    blocks = chunk.split_blocks("T\n## A\nline one\n## \nline two\n## B\nline three")
    assert [b["heading"] for b in blocks] == ["Overview", "A", "B"]
    assert blocks[1]["lines"] == ["line one", "line two"]


def test_section_strategy_keeps_subsections_with_parent_and_labels_every_chunk():
    chunks = chunk.chunk_docs([PROGRAM], strategy="section", max_words=200)
    sections = [c["section"] for c in chunks]
    assert "Term 1" in sections and "Term 2" in sections
    term1 = next(c for c in chunks if c["section"] == "Term 1")
    assert "MATH 141* - Calculus I 4 Credit Hours" in term1["text"]
    assert "Term 1 Milestone Notes" in term1["text"]          # subsection stays with its parent
    assert "COSC 203" not in term1["text"]                    # next section does not leak in
    for c in chunks:
        assert c["url"] == PROGRAM["url"] and c["title"] == PROGRAM["title"] and c["section"]
        assert c["text"].startswith("Fake Major, BS")         # context header for retrieval
        assert c["doc_type"] == "program"
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_tiny_section_is_merged_forward_not_emitted_alone():
    chunks = chunk.chunk_docs([PROGRAM], strategy="section", max_words=200)
    assert all(c["section"] != "Campus: Knoxville" for c in chunks)
    assert any("Campus: Knoxville" in c["text"] for c in chunks)


def test_no_source_line_is_lost():
    for strategy in chunk.STRATEGIES:
        chunks = chunk.chunk_docs([PROGRAM], strategy=strategy, max_words=40)
        joined = "\n".join(c["text"] for c in chunks)
        for line in body_lines(PROGRAM_TEXT):
            assert line in joined, (strategy, line)


def test_oversize_section_splits_at_subheadings_before_splitting_lines():
    chunks = chunk.chunk_docs([PROGRAM], strategy="section", max_words=25)
    sections = [c["section"] for c in chunks]
    assert "Term 1 > Term 1 Milestone Notes" in sections


def test_oversize_leaf_section_is_split_by_lines_with_overlap():
    lines = [f"MATH {100 + i} - Course number {i} 3 Credit Hours" for i in range(30)]
    doc = {**PROGRAM, "text": "Big Major\n## Electives\n" + "\n".join(lines)}
    chunks = chunk.chunk_docs([doc], strategy="section", max_words=60, overlap_lines=2)
    assert len(chunks) > 2
    for c in chunks:
        assert chunk.n_words(c["body"]) <= 60
        assert c["section"] == "Electives"
    for a, b in zip(chunks, chunks[1:]):
        assert a["body"].split("\n")[-2:] == b["body"].split("\n")[:2]


def test_packed_strategy_merges_small_sections_up_to_max_words_with_overlap():
    section = chunk.chunk_docs([PROGRAM], strategy="section", max_words=200)
    packed = chunk.chunk_docs([PROGRAM], strategy="packed", max_words=200, overlap_sections=0)
    assert len(packed) < len(section)
    # six equal sections of 10 words; the cap fits three sections plus headings
    text = "Even Major\n" + "\n".join(
        f"## Part {i}\ncourse {i} line with exactly ten words in it here ok" for i in range(6))
    overlapped = chunk.chunk_docs([{**PROGRAM, "text": text}], strategy="packed",
                                  max_words=40, overlap_sections=1)
    assert len(overlapped) >= 2
    for a, b in zip(overlapped, overlapped[1:]):
        assert a["body"].split("\n")[-1] in b["body"]      # last section repeated
        assert chunk.n_words(b["body"]) <= 40


def test_packed_drops_overlap_when_it_would_not_fit():
    overlapped = chunk.chunk_docs([PROGRAM], strategy="packed", max_words=60, overlap_sections=1)
    assert len(overlapped) >= 2
    assert all(chunk.n_words(c["body"]) <= 60 for c in overlapped)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk.chunk_docs([PROGRAM], strategy="nope")
