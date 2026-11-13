"""Scoring maths for the benchmark. Pure functions: no files, no API calls.

Retrieval metrics ask "did the search find a page that holds the answer".
Answer metrics are the ones GRADING_RUBRIC.md defined before any results
existed; the formulas here must stay word-for-word faithful to that file.
"""
from __future__ import annotations

from collections import Counter

LABELS = ("correct", "partially_correct", "incorrect",
          "correct_refusal", "wrong_refusal", "hallucinated")
REFUSALS = ("correct_refusal", "wrong_refusal")
HIT_KS = (1, 3, 5)
PARTIAL_CREDIT = 0.5
TOKENS_PER_MTOK = 1_000_000


def _ratio(numerator: float, denominator: float) -> float | None:
    """None, not 0, when there is nothing to divide by: "no refusals happened"
    is different from "every refusal was wrong"."""
    return numerator / denominator if denominator else None


def first_hit_rank(retrieved_urls: list[str], source_urls: list[str]) -> int | None:
    """1-based position of the first retrieved chunk that comes from one of the
    question's source pages, or None when none of them was retrieved."""
    sources = set(source_urls)
    for rank, url in enumerate(retrieved_urls, start=1):
        if url in sources:
            return rank
    return None


def source_recall(retrieved_urls: list[str], source_urls: list[str]) -> float:
    """Share of the question's source pages that appear among the retrieved
    chunks. Matters for multi-hop questions, which need more than one page."""
    sources = set(source_urls)
    return len(sources & set(retrieved_urls)) / len(sources)


def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...] = HIT_KS) -> dict:
    """Hit rate @k (share of questions with a source page in the top k) and
    MRR (mean of 1/rank of the first source page, 0 when it was missed)."""
    n = len(ranks)
    out: dict = {"n": n}
    for k in ks:
        out[f"hit@{k}"] = _ratio(sum(1 for r in ranks if r is not None and r <= k), n)
    out["mrr"] = _ratio(sum(1 / r for r in ranks if r is not None), n)
    return out


def answer_metrics(grades: list[dict]) -> dict:
    """The rubric's metrics from a list of {label, cited, citation_correct}."""
    counts = Counter(g["label"] for g in grades)
    unknown = set(counts) - set(LABELS)
    if unknown:
        raise ValueError(f"labels not in the rubric: {sorted(unknown)}")
    n = len(grades)
    right = counts["correct"] + counts["correct_refusal"]
    attempts = [g for g in grades if g["label"] not in REFUSALS]
    cited = [g for g in attempts if g["cited"] == "yes"]
    return {
        "n": n,
        "accuracy": _ratio(right, n),
        "partial_accuracy": _ratio(right + PARTIAL_CREDIT * counts["partially_correct"], n),
        "hallucination_rate": _ratio(counts["hallucinated"], n),
        "refusal_precision": _ratio(counts["correct_refusal"],
                                    counts["correct_refusal"] + counts["wrong_refusal"]),
        "citation_rate": _ratio(len(cited), len(attempts)),
        "citation_accuracy": _ratio(sum(1 for g in cited if g["citation_correct"] == "yes"),
                                    len(cited)),
        "counts": {label: counts[label] for label in LABELS},
    }


def failure_group(category: str, label: str, retrieved: bool) -> str | None:
    """Which kind of failure a graded answer is, for error analysis. None for
    a right answer. `retrieved` says whether a source page was in the context.

    Checked in this order, so each failure lands in exactly one group:
    hallucination, bad_refusal (refused an answerable question, or failed to
    refuse an unanswerable one), retrieval_miss, wrong_despite_retrieval."""
    if label in ("correct", "correct_refusal"):
        return None
    if label == "hallucinated":
        return "hallucination"
    if label == "wrong_refusal" or category == "unanswerable":
        return "bad_refusal"
    return "wrong_despite_retrieval" if retrieved else "retrieval_miss"


def cost_usd(input_tokens: int, output_tokens: int,
             price_per_mtok: tuple[float, float] | None) -> float | None:
    """Dollar cost of one call from (input, output) prices per million tokens.
    None when the model's price is not in config."""
    if price_per_mtok is None:
        return None
    price_in, price_out = price_per_mtok
    return (input_tokens * price_in + output_tokens * price_out) / TOKENS_PER_MTOK
