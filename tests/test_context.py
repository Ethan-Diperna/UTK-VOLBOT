from src import context


def course(n, code, prereq=""):
    text = f"{code} - Title\nCredit Hours: 3" + (f"\n(RE) Prerequisite(s): {prereq}" if prereq else "")
    return {"id": f"course:{n}", "doc_id": f"course:{n}", "doc_type": "course", "code": code,
            "url": f"u/c{n}", "title": f"{code} - Title", "section": "", "text": text}


def section(page, i):
    return {"id": f"program:{page}#{i:02d}", "doc_id": f"program:{page}", "doc_type": "program", "code": "",
            "url": f"u/p{page}", "title": f"Program {page}", "section": f"Term {i}", "text": f"p{page} term {i}"}


CHUNKS = {c["id"]: c for c in [
    course(1, "ECE 415", "ECE 316 with a grade of C or better."),
    course(2, "ECE 316", "ECE 315 with a grade of C or better."),
    course(3, "ECE 315", "ECE 202 with a grade of C or better."),
    course(4, "ECE 202"),
    course(5, "MATH 251", "MATH 142* or MATH 148*"),
    course(6, "MATH 142"),
    course(7, "MATH 148"),
    section(10, 0), section(10, 1), section(10, 2),
    section(20, 0), section(20, 1),
    section(30, 0), section(30, 1),
]}


def hit(cid, rank):
    return {**CHUNKS[cid], "rank": rank, "score": 1.0, "mode": "hybrid_rerank"}


def ids(chunks):
    return [c["id"] for c in chunks]


def test_course_codes_are_found_in_any_spelling_once_each_in_order():
    assert context.course_codes("is cosc360 like COSC 367, or cosc-360?") == ["COSC 360", "COSC 367"]
    assert context.course_codes("which class covers machine learning") == []


def test_no_steps_returns_the_search_hits_unchanged():
    hits = [hit("course:5", 1), hit("program:10#01", 2)]
    assert context.build_context("MATH 251?", hits, CHUNKS, steps=()) == hits


def test_named_courses_puts_every_course_written_in_the_question_first():
    hits = [hit("course:7", 1)]
    out = context.build_context("difference between math 251 and MATH 148", hits, CHUNKS, steps=("named_courses",))
    assert ids(out) == ["course:5", "course:7"]          # MATH 251 added, MATH 148 not duplicated
    assert out[0]["added_by"] == "named_courses" and out[1]["added_by"] == "search"


def test_a_code_with_no_catalog_entry_adds_nothing():
    hits = [hit("course:5", 1)]
    out = context.build_context("how many hours is COSC 302", hits, CHUNKS, steps=("named_courses",))
    assert ids(out) == ["course:5"]


def test_program_pages_replaces_a_section_hit_with_its_whole_page_in_page_order():
    hits = [hit("program:10#02", 1), hit("course:5", 2), hit("program:10#00", 3)]
    out = context.build_context("q", hits, CHUNKS, steps=("program_pages",))
    assert ids(out) == ["program:10#00", "program:10#01", "program:10#02", "course:5"]


def test_program_pages_expands_only_the_best_ranked_pages_up_to_the_limit():
    hits = [hit("program:10#01", 1), hit("program:20#00", 2), hit("program:30#01", 3)]
    out = context.build_context("q", hits, CHUNKS, steps=("program_pages",), max_pages=2)
    assert ids(out) == ["program:10#00", "program:10#01", "program:10#02",
                        "program:20#00", "program:20#01", "program:30#01"]


def test_prerequisites_follows_the_chain_from_a_named_course():
    out = context.build_context("full prerequisite chain for ECE 415", [hit("course:1", 1)], CHUNKS,
                                steps=("prerequisites",))
    assert ids(out) == ["course:1", "course:2", "course:3", "course:4"]
    assert out[1]["added_by"] == "prerequisites"


def test_prerequisites_stops_at_the_hop_limit_and_the_chunk_limit():
    hits = [hit("course:1", 1)]
    one_hop = context.build_context("ECE 415 chain", hits, CHUNKS, steps=("prerequisites",), max_hops=1)
    capped = context.build_context("ECE 415 chain", hits, CHUNKS, steps=("prerequisites",), max_added=2)
    assert ids(one_hop) == ["course:1", "course:2"]
    assert ids(capped) == ["course:1", "course:2", "course:3"]


def test_prerequisites_includes_both_sides_of_an_or():
    out = context.build_context("what do i need before MATH 251", [], CHUNKS, steps=("prerequisites",))
    assert ids(out) == ["course:6", "course:7"]          # MATH 251 itself comes from search or named_courses


def test_all_steps_together_never_repeat_a_chunk():
    hits = [hit("course:2", 1), hit("program:10#01", 2)]
    out = context.build_context("ECE 415 and ece 316", hits, CHUNKS, steps=context.STEPS)
    assert len(ids(out)) == len(set(ids(out)))
    assert ids(out)[:2] == ["course:1", "course:2"]
