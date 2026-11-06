from collections import Counter

from benchmark import lock, questions
from benchmark.split import stratified_split


def make_questions(n_per_cat: int) -> list[dict]:
    out = []
    for cat in questions.CATEGORIES:
        for i in range(n_per_cat):
            urls = [] if cat == "unanswerable" else [f"https://catalog.utk.edu/x?coid={i}"]
            out.append({"id": f"{cat[:3]}{i:03d}", "question": f"q {cat} {i}",
                        "expected_answer": "a", "source_urls": urls,
                        "category": cat, "origin": "self", "notes": ""})
    return out


def test_validate_accepts_well_formed_set():
    assert questions.validate(make_questions(3)) == []


def test_validate_rejects_bad_category_duplicate_id_and_missing_source():
    qs = make_questions(1)
    qs[0]["category"] = "nope"
    qs[1]["source_urls"] = []
    qs.append(dict(qs[2]))
    problems = questions.validate(qs)
    assert any("bad category" in p for p in problems)
    assert any("needs at least one source url" in p for p in problems)
    assert any("duplicate id" in p for p in problems)


def test_split_is_stratified_and_disjoint():
    qs = make_questions(10)
    dev, test = stratified_split(qs, seed=1)
    assert len(dev) == 24 and len(test) == 36
    assert Counter(q["category"] for q in dev) == {c: 4 for c in questions.CATEGORIES}
    assert {q["id"] for q in dev}.isdisjoint({q["id"] for q in test})
    assert {q["id"] for q in dev} | {q["id"] for q in test} == {q["id"] for q in qs}


def test_split_is_deterministic_for_a_seed():
    qs = make_questions(5)
    assert stratified_split(qs, seed=7) == stratified_split(qs, seed=7)
    assert stratified_split(qs, seed=7) != stratified_split(qs, seed=8)


def test_lock_verifies_and_detects_tampering(tmp_path):
    test_path = tmp_path / "questions_test.jsonl"
    lock_path = tmp_path / "test_set.sha256"
    questions.save(test_path, make_questions(2))
    digest = lock.write_lock(test_path, lock_path)
    assert len(digest) == 64
    assert lock.read_lock(lock_path)["count"] == "12"
    assert lock.verify(test_path, lock_path)
    qs = questions.load(test_path)
    qs[0]["expected_answer"] = "changed"
    questions.save(test_path, qs)
    assert not lock.verify(test_path, lock_path)
