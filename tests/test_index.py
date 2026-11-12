import pytest

from src import index


def make_chunk(cid, text, **over):
    base = {"id": cid, "doc_id": cid, "url": f"https://catalog.utk.edu/x?id={cid}",
            "title": text.split("\n")[0], "section": "", "doc_type": "course",
            "dept": "COSC", "code": "", "catalog_year": "2026-2027", "text": text}
    return {**base, **over}


CHUNKS = [
    make_chunk("c302", "COSC 302 - Data Structures and Algorithms II\nCredit Hours: 4", code="COSC 302"),
    make_chunk("c202", "COSC 202 - Data Structures and Algorithms I\nCredit Hours: 4", code="COSC 202"),
    make_chunk("m141", "MATH 141 - Calculus I\nCredit Hours: 4", code="MATH 141", dept="MATH"),
    make_chunk("e201", "ECE 201 - Circuits I\nPrerequisite(s): MATH 142", code="ECE 201", dept="ECE"),
]


@pytest.mark.parametrize("variant", ["COSC 302", "COSC302", "cosc 302", "cosc302", "Cosc-302"])
def test_tokenize_gives_same_tokens_for_every_spelling_of_a_course_code(variant):
    assert index.tokenize(variant) == index.tokenize("COSC 302")
    assert "cosc302" in index.tokenize(variant)


def test_tokenize_keeps_code_parts_so_subject_alone_still_matches():
    tokens = index.tokenize("What are the prerequisites for COSC 302?")
    assert {"cosc", "302", "cosc302", "prerequisites"} <= set(tokens)


def test_tokenize_lowercases_and_drops_punctuation():
    assert index.tokenize("Credit Hours: 4 (RE) Prerequisite(s)") == [
        "credit", "hours", "4", "re", "prerequisite", "s"]


@pytest.mark.parametrize("query", ["COSC 302", "COSC302", "cosc 302", "whats the prereq for cosc302"])
def test_bm25_ranks_the_exact_course_first_for_any_code_spelling(query):
    bm25 = index.Bm25Index.build(CHUNKS)
    assert bm25.search(query, k=2)[0][0] == "c302"


def test_bm25_search_returns_ids_with_scores_best_first_and_skips_zero_scores():
    bm25 = index.Bm25Index.build(CHUNKS)
    hits = bm25.search("calculus", k=4)
    assert [h[0] for h in hits] == ["m141"]
    assert hits[0][1] > 0


def test_bm25_save_and_load_round_trip(tmp_path):
    path = tmp_path / "bm25.json"
    index.Bm25Index.build(CHUNKS).save(path, meta={"strategy": "section"})
    loaded = index.Bm25Index.load(path)
    assert loaded.ids == [c["id"] for c in CHUNKS]
    assert loaded.meta["strategy"] == "section"
    assert loaded.search("ECE 201", k=1)[0][0] == "e201"


def test_build_chroma_stores_every_chunk_with_metadata_and_finds_nearest(tmp_path):
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.7, 0.0]]
    col = index.build_chroma(CHUNKS, embeddings, tmp_path / "chroma", meta={"embedding_model": "fake"})
    assert col.count() == len(CHUNKS)
    res = col.query(query_embeddings=[[0.0, 0.1, 0.99]], n_results=1)
    assert res["ids"][0] == ["m141"]
    assert res["metadatas"][0][0]["url"] == CHUNKS[2]["url"]
    assert res["metadatas"][0][0]["code"] == "MATH 141"
    assert res["documents"][0][0] == CHUNKS[2]["text"]


def test_build_chroma_rebuild_replaces_old_contents(tmp_path):
    index.build_chroma(CHUNKS, [[1.0, 0.0]] * 4, tmp_path / "chroma", meta={})
    col = index.build_chroma(CHUNKS[:2], [[1.0, 0.0]] * 2, tmp_path / "chroma", meta={})
    assert col.count() == 2


def test_build_chroma_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        index.build_chroma(CHUNKS, [[1.0, 0.0]], tmp_path / "chroma", meta={})
