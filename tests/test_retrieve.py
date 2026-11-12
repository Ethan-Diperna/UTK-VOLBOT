import pytest

from src import retrieve
from src.index import Bm25Index


def make_chunk(cid, text, code=""):
    return {"id": cid, "doc_id": cid, "url": f"https://catalog.utk.edu/x?id={cid}",
            "title": text.split("\n")[0], "section": "", "doc_type": "course",
            "code": code, "text": text}


CHUNKS = [
    make_chunk("c302", "COSC 302 - Data Structures and Algorithms II\nCredit Hours: 4", "COSC 302"),
    make_chunk("c312", "COSC 312 - Algorithm Analysis and Automata\nCredit Hours: 3", "COSC 312"),
    make_chunk("c366", "COSC 366 - Introduction to Cybersecurity\nSecurity goals and threat modeling", "COSC 366"),
    make_chunk("m141", "MATH 141 - Calculus I\nCredit Hours: 4", "MATH 141"),
    make_chunk("e201", "ECE 201 - Circuits I\nOhm's law", "ECE 201"),
]
VECTOR_ORDER = ["c366", "c312", "m141", "e201", "c302"]   # what the fake embedder "thinks"


def fake_vector(query, k):
    return [(cid, 1.0 - i * 0.1) for i, cid in enumerate(VECTOR_ORDER[:k])]


def fake_rerank(query, texts):
    # pretend the cross-encoder loves anything mentioning Calculus, then Circuits
    return [10.0 if "Calculus" in t else 5.0 if "Circuits" in t else 0.0 for t in texts]


@pytest.fixture
def retriever():
    return retrieve.Retriever({c["id"]: c for c in CHUNKS}, Bm25Index.build(CHUNKS),
                              fake_vector, fake_rerank)


def test_rrf_score_is_sum_of_reciprocal_ranks():
    fused = dict(retrieve.rrf([["a", "b"], ["a", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_item_in_both_lists_beats_item_first_in_only_one():
    fused = retrieve.rrf([["top_of_one", "shared"], ["other", "shared"]], k=60)
    assert fused[0][0] == "shared"


def test_rrf_is_best_first_and_breaks_ties_by_first_appearance():
    fused = retrieve.rrf([["a", "b"], ["c", "d"]], k=60)
    assert [cid for cid, _ in fused] == ["a", "c", "b", "d"]


def test_rrf_handles_empty_rankings():
    assert retrieve.rrf([[], []]) == []
    assert [cid for cid, _ in retrieve.rrf([["a"], []])] == ["a"]


def test_vector_mode_returns_vector_order(retriever):
    hits = retriever.search("anything", mode="vector", k=3)
    assert [h["id"] for h in hits] == VECTOR_ORDER[:3]


def test_bm25_mode_finds_exact_course_code(retriever):
    hits = retriever.search("cosc302 credit hours", mode="bm25", k=2)
    assert hits[0]["id"] == "c302"


def test_hybrid_mode_lifts_a_chunk_both_searches_agree_on(retriever):
    # vector ranks c312 second; BM25 ranks it first for this query
    hits = retriever.search("COSC 312 algorithm", mode="hybrid", k=3)
    assert hits[0]["id"] == "c312"


def test_hybrid_rerank_mode_reorders_candidates_by_cross_encoder(retriever):
    hits = retriever.search("COSC 312 algorithm", mode="hybrid_rerank", k=2, candidates=5)
    assert [h["id"] for h in hits] == ["m141", "e201"]
    assert hits[0]["score"] == 10.0


def test_hybrid_rerank_only_scores_the_candidate_pool(retriever):
    seen = []

    def spy(query, texts):
        seen.append(len(texts))
        return [0.0] * len(texts)

    r = retrieve.Retriever(retriever.chunks, retriever.bm25, fake_vector, spy)
    r.search("COSC 312 algorithm", mode="hybrid_rerank", k=1, candidates=2)
    assert seen == [2]


def test_results_carry_citation_fields_rank_and_mode(retriever):
    hits = retriever.search("calculus", mode="hybrid", k=2)
    assert [h["rank"] for h in hits] == [1, 2]
    for h in hits:
        assert h["mode"] == "hybrid"
        assert h["url"].startswith("https://catalog.utk.edu/") and h["title"] and h["text"]
        assert isinstance(h["score"], float)


def test_k_limits_results_in_every_mode(retriever):
    for mode in retrieve.MODES:
        assert len(retriever.search("COSC 302 calculus circuits", mode=mode, k=2)) == 2


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="psychic")


def test_empty_query_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")
