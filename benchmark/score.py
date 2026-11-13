"""Un-blind the grades and score the head-to-head on the test set.

  python -m benchmark.score

Run only after grading is finished. It reads the answer key (checked against
the hash committed before grading began), turns "Response A/B" grades back
into per-system grades, and reports for each system: accuracy with a 95%
bootstrap interval, accuracy by category, hallucination rate, and citation
rate and accuracy. The two systems are compared with a paired bootstrap
interval for the accuracy gap and McNemar's exact test. If the LLM judge has
graded the same sheet (benchmark.judge_sheet), its numbers are reported next
to the human ones together with how often the two graders agree.

Citations are scored by rule, not by eye, because showing them would reveal
the system: a response "cites correctly" when one of its cited pages is a
source page of the question, matched by catalog page id or by page title.
Only answerable questions count, and only responses whose cited sources were
recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from benchmark import metrics, stats
from benchmark.collect_baseline import load_responses, responses_path
from benchmark.grade import GRADES_PATH, load_grades
from benchmark.judge_sheet import JUDGE_GRADES_PATH
from benchmark.make_blind_sheet import GRADING_DIR, KEY_PATH, latest_test_run
from benchmark.questions import TEST_PATH, load
from src import config as cfg
from src.chunk import load_jsonl

KEY_HASH_PATH = GRADING_DIR / "blind_key.sha256"
SYSTEMS = ("ours", "baseline")
RIGHT = ("correct", "correct_refusal")
SUMMARY_KEYS = ("n", "accuracy", "partial_accuracy", "hallucination_rate", "refusal_precision", "counts")
SITE_SUFFIX_RE = re.compile(r"\s+-\s+(?:-\s+)?(?:UTK Catalog|Undergraduate Catalog|"
                            r"University of Tennessee, Knoxville)\s*$", re.I)
MIN_TRUNCATED_TITLE_CHARS = 10


def load_key(key_path: Path = KEY_PATH, hash_path: Path = KEY_HASH_PATH) -> dict[str, dict[str, str]]:
    """The A/B assignment, refused unless the key file is byte-identical to
    the one whose hash was committed before grading."""
    expected = hash_path.read_text(encoding="utf-8").strip()
    if hashlib.sha256(key_path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"{key_path.name} does not match {hash_path.name}; refusing to un-blind.")
    return json.loads(key_path.read_text(encoding="utf-8"))["assignments"]


def unblind(grades: dict[str, dict], key: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """{system: {question id: label}} from grades keyed by item ("q1-A")."""
    missing = [f"{qid}-{label}" for qid in key for label in ("A", "B") if f"{qid}-{label}" not in grades]
    if missing:
        raise ValueError(f"no grade for: {', '.join(missing)}")
    by_system: dict[str, dict[str, str]] = {name: {} for name in SYSTEMS}
    for qid, assignment in key.items():
        for label, system in assignment.items():
            by_system[system][qid] = grades[f"{qid}-{label}"]["label"]
    return by_system


def _as_grades(labels: list[str]) -> list[dict]:
    return [{"label": label, "cited": "no", "citation_correct": "na"} for label in labels]


def system_summary(labels: dict[str, str], questions: list[dict]) -> dict:
    """Rubric metrics for one system, a 95% interval for accuracy, and
    accuracy per question category."""
    ordered = [labels[q["id"]] for q in questions]
    overall = metrics.answer_metrics(_as_grades(ordered))
    by_category = {}
    for category in sorted({q["category"] for q in questions}):
        subset = metrics.answer_metrics(_as_grades([labels[q["id"]] for q in questions if q["category"] == category]))
        by_category[category] = {"n": subset["n"], "accuracy": subset["accuracy"]}
    return {**{key: overall[key] for key in SUMMARY_KEYS},
            "accuracy_ci": list(stats.bootstrap_ci([float(label in RIGHT) for label in ordered])),
            "by_category": by_category}


def compare(ours: dict[str, str], baseline: dict[str, str], questions: list[dict]) -> dict:
    ours_right = [ours[q["id"]] in RIGHT for q in questions]
    base_right = [baseline[q["id"]] in RIGHT for q in questions]
    diff, low, high = stats.paired_bootstrap_diff([float(x) for x in ours_right], [float(x) for x in base_right])
    return {"accuracy_difference": diff, "difference_ci": [low, high],
            "mcnemar": stats.mcnemar_exact(ours_right, base_right)}


def grader_agreement(first: dict[str, dict], second: dict[str, dict]) -> dict:
    """Agreement between two graders over the items both graded: on the exact
    label, and on the coarser right / not right split the accuracy uses."""
    items = sorted(set(first) & set(second))
    a = [first[i]["label"] for i in items]
    b = [second[i]["label"] for i in items]
    a_right = [str(label in RIGHT) for label in a]
    b_right = [str(label in RIGHT) for label in b]
    return {"n": len(items), "exact_agreement": stats.agreement(a, b), "kappa": stats.cohen_kappa(a, b),
            "right_wrong_agreement": stats.agreement(a_right, b_right),
            "right_wrong_kappa": stats.cohen_kappa(a_right, b_right)}


def _title_key(title: str) -> str:
    title = re.sub(r"^\s*Program:\s*", "", title, flags=re.I)
    title = SITE_SUFFIX_RE.sub("", title).replace("...", "").replace("…", "")
    return re.sub(r"[^a-z0-9]", "", title.lower())


def titles_match(cited_title: str, source_title: str) -> bool:
    """Same page title, ignoring the site name the search result adds. A cited
    title cut short with "..." matches when it is the start of the source title."""
    cited, source = _title_key(cited_title), _title_key(source_title)
    if not cited or not source:
        return False
    if cited == source:
        return True
    truncated = "..." in cited_title or "…" in cited_title
    return truncated and len(cited) >= MIN_TRUNCATED_TITLE_CHARS and source.startswith(cited)


def _page_key(url: str) -> tuple | str:
    """Catalog pages are identified by script name plus ids, so extra query
    parameters ("&returnto=...", "&print") do not make a different page."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    ids = tuple(query.get(name, [""])[0] for name in ("catoid", "coid", "poid"))
    return (parsed.netloc, parsed.path, *ids) if any(ids[1:]) else url


def citation_fields(question: dict, cited: list[dict], titles: dict[str, str]) -> dict[str, str]:
    """The rubric's two citation fields for one response. `cited` holds the
    sources the response cited as {title, url}; `titles` maps a catalog URL to
    its page title."""
    if not cited:
        return {"cited": "no", "citation_correct": "na"}
    source_keys = {_page_key(u) for u in question["source_urls"]}
    source_titles = [titles[u] for u in question["source_urls"] if u in titles]
    hit = any(_page_key(c["url"]) in source_keys or any(titles_match(c.get("title", ""), t) for t in source_titles)
              for c in cited)
    return {"cited": "yes", "citation_correct": "yes" if hit else "no"}


def citation_summary(labels: dict[str, str], cited_by_question: dict[str, list[dict]],
                     questions: list[dict], titles: dict[str, str]) -> dict:
    """Citation rate and accuracy over answerable questions the system tried
    to answer. A question missing from `cited_by_question` has no recorded
    sources and is counted separately instead of being scored as uncited."""
    answerable = [q for q in questions if q["source_urls"]]
    known = [q for q in answerable if q["id"] in cited_by_question]
    scored = [q for q in known if labels[q["id"]] not in metrics.REFUSALS]
    fields = [citation_fields(q, cited_by_question[q["id"]], titles) for q in scored]
    cited = [f for f in fields if f["cited"] == "yes"]
    return {"n_scored": len(scored), "n_without_citation_data": len(answerable) - len(known),
            "citation_rate": len(cited) / len(scored) if scored else None,
            "citation_accuracy": (sum(f["citation_correct"] == "yes" for f in cited) / len(cited)) if cited else None}


def grader_report(grades: dict[str, dict], key: dict, questions: list[dict],
                  cited: dict[str, dict[str, list[dict]]], titles: dict[str, str]) -> dict:
    by_system = unblind(grades, key)
    report = {name: {**system_summary(by_system[name], questions),
                     "citations": citation_summary(by_system[name], cited[name], questions, titles)}
              for name in SYSTEMS}
    return {**report, "comparison": compare(by_system["ours"], by_system["baseline"], questions)}


def _pct(value: float | None) -> str:
    return "   -  " if value is None else f"{100 * value:5.1f}%"


def format_report(report: dict) -> str:
    lines = []
    for grader in ("human", "judge"):
        if grader not in report["graders"]:
            continue
        g = report["graders"][grader]
        lines += ["", f"=== {grader} grades, {report['n_questions']} test questions ===",
                  f"{'':<22}{'ours':>18}{'baseline':>18}"]
        for label, key in (("accuracy", "accuracy"), ("partial-credit acc.", "partial_accuracy"),
                           ("hallucination rate", "hallucination_rate")):
            lines.append(f"{label:<22}{_pct(g['ours'][key]):>18}{_pct(g['baseline'][key]):>18}")
        cis = [f"{_pct(g[s]['accuracy_ci'][0])} to {_pct(g[s]['accuracy_ci'][1])}" for s in SYSTEMS]
        lines.append(f"{'accuracy 95% CI':<22}{cis[0]:>18}{cis[1]:>18}")
        for category in g["ours"]["by_category"]:
            cells = [f"{_pct(g[s]['by_category'][category]['accuracy'])} (n={g[s]['by_category'][category]['n']})"
                     for s in SYSTEMS]
            lines.append(f"{'  ' + category:<22}{cells[0]:>18}{cells[1]:>18}")
        for label, key in (("citation rate", "citation_rate"), ("citation accuracy", "citation_accuracy")):
            cells = [f"{_pct(g[s]['citations'][key])} (n={g[s]['citations']['n_scored']})" for s in SYSTEMS]
            lines.append(f"{label:<22}{cells[0]:>18}{cells[1]:>18}")
        c = g["comparison"]
        lines += [f"accuracy gap (ours - baseline): {_pct(c['accuracy_difference'])}, "
                  f"95% CI {_pct(c['difference_ci'][0])} to {_pct(c['difference_ci'][1])}",
                  f"McNemar: ours right / baseline wrong on {c['mcnemar']['ours_only']}, the reverse on "
                  f"{c['mcnemar']['baseline_only']}, p = {c['mcnemar']['p_value']:.4f}"]
    if "agreement" in report:
        a = report["agreement"]
        lines += ["", f"human vs judge over {a['n']} responses: same label {_pct(a['exact_agreement'])} "
                      f"(kappa {a['kappa']:.2f}), same right/not-right {_pct(a['right_wrong_agreement'])} "
                      f"(kappa {a['right_wrong_kappa']:.2f})"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Un-blind and score the head-to-head.")
    ap.add_argument("--run", type=Path, help="test run folder (default: the newest *_test run)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    questions = load(TEST_PATH)
    key = load_key()
    run_dir = a.run or latest_test_run()
    titles = {d["url"]: d["title"] for d in load_jsonl(cfg.DOCS_PATH)}
    cited = {
        "ours": {r["id"]: r.get("citations", []) for r in load_jsonl(run_dir / "rows.jsonl")},
        "baseline": {qid: row["sources"] for qid, row in load_responses(responses_path("test")).items()
                     if "sources" in row},
    }
    graders = {"human": load_grades(GRADES_PATH)}
    if JUDGE_GRADES_PATH.exists():
        graders["judge"] = load_grades(JUDGE_GRADES_PATH)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "n_questions": len(questions), "test_run": run_dir.name,
              "bootstrap": {"resamples": stats.BOOTSTRAP_RESAMPLES, "seed": stats.BOOTSTRAP_SEED,
                            "level": stats.CONFIDENCE_LEVEL},
              "graders": {name: grader_report(g, key, questions, cited, titles) for name, g in graders.items()}}
    if "judge" in graders:
        report["agreement"] = grader_agreement(graders["human"], graders["judge"])
    print(format_report(report))
    out_dir = cfg.RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_headtohead"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
