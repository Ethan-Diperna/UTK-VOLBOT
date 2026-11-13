# Tuning log (dev set only)

Everything here was measured on `questions_dev.jsonl` (44 questions, 38 answerable).
The test set was not read or run at any point. Answer grades are from the LLM judge
(`claude-sonnet-5-5`, rubric sent verbatim), not from a human. Run folders are under
`benchmark/results/`; each holds its full config and one row per question.

With 44 questions, one question is 2.3 points of accuracy. Two runs of the same
config differed by one grade (s029, s019) because Claude's answers are not
deterministic. Read differences of one or two questions as noise.

## 1. Retrieval: mode and chunking (2026-11-12, run `20261113T002541Z_dev`, free)

Was a chunk from a source page retrieved in the top k? hit@5 by mode and chunking:

| Chunking | vector | bm25 | hybrid | hybrid_rerank |
|---|---|---|---|---|
| section:250 (default) | 0.92 | 0.76 | 0.87 | 0.95 |
| section:150 | 0.92 | 0.76 | 0.92 | 0.95 |
| packed:250 | 0.89 | 0.87 | 0.84 | 0.92 |
| packed:350 | 0.84 | 0.84 | 0.95 | 0.89 |

- `hybrid_rerank` has the best MRR in every chunking (0.80 to 0.85). Kept as default.
- Chunking differences are one to three questions. No evidence to move off `section:250`. Kept.
- **Did not work as expected:** plain hybrid (RRF) is below vector search alone on the
  default chunking (hit@5 0.87 vs 0.92, MRR 0.71 vs 0.74). BM25 is the weak half on
  casually worded questions and drags the fusion down. Hybrid only pays off once the
  cross-encoder re-scores the fused candidates.
- `packed:350` reaches 495 tokens per chunk against the embedding model's 512 limit,
  and the reranker reads question plus chunk in 512, so its numbers include truncation.

## 2. Answers by retrieval mode, no context steps (run `20261113T010822Z_dev`)

| Mode | Accuracy | Partial-credit | Hallucinated | Wrong refusals | Retrieval ms |
|---|---|---|---|---|---|
| vector | 0.64 | 0.74 | 0 | 7 | 10 |
| bm25 | 0.48 | 0.59 | 0 | 13 | <1 |
| hybrid | 0.59 | 0.70 | 1 | 7 | 12 |
| hybrid_rerank | 0.70 | 0.78 | 0 | 4 | 222 |

Generation: about 3.8 s and $0.015 per question (Opus 5.5); judging adds about $0.010.

## 3. Error analysis of the 13 `hybrid_rerank` failures

No hallucinations. Every failure was missing context: the model said, correctly, that
the excerpts did not hold the answer.

| Root cause | Count | Questions |
|---|---|---|
| Answer spread over many sections of one program page | 6 | s010, s011, s020, s026, s027, s029 |
| A course or program named in the question was not retrieved | 3 | cd061, cd063, s013 |
| Prerequisite chain needs a second lookup | 2 | cd053, cd054 |
| Other (list question missing one course; DATA 101 vs DATA 111) | 2 | s023, s028 |

Metric weakness found: hit@k counts a hit when any chunk of the right page is
retrieved, even if it is the wrong section (s027: right page at rank 2, answer still
incomplete). It overstates retrieval for program pages.

## 4. Fix: context steps (`src/context.py`)

Three generic steps between search and generation. Each run alone, then together,
on `hybrid_rerank`:

| Context steps | Accuracy | Partial-credit | Hallucinated | Chunks sent | Answer cost | Grades changed vs none |
|---|---|---|---|---|---|---|
| none | 0.70 (31/44) | 0.78 | 0 | 5 | $0.015 | |
| named_courses | 0.73 (32/44) | 0.80 | 0 | 5.0 | $0.015 | cd061 fixed |
| prerequisites | 0.73 (32/44) | 0.80 | 0 | 7.3 | $0.018 | cd053 fixed |
| program_pages | 0.80 (35/44) | 0.88 | 0 | 17.7 | $0.034 | s010, s020, s026, s027 fixed; s029 half fixed |
| all three | 0.84 (37/44) | 0.90 | 0 | 20.0 | $0.037 | 7 fixed, 1 worse (s019) |

- `program_pages` gives most of the gain and most of the cost: answer cost per question
  goes from $0.015 to $0.037 and generation from 3.8 s to 4.5 s.
- `named_courses` and `prerequisites` fix one question each. That is within noise on
  its own; they are kept because they are cheap and the failure they repair is real.
- **Overfitting warning:** these steps were designed after reading these same dev
  failures. 0.84 is an optimistic number. The test set is the honest one.

## 5. Still failing on dev with all steps on (7)

| Question | Why |
|---|---|
| s011 "which math classes do CS majors gotta take" | "CS" never matches the Computer Science page; search returns the Mathematics major. An abbreviation map would fix it. Not added: one question is thin evidence. |
| s028 intro DATA class | Search finds DATA 111 (majors version), not DATA 101. |
| cd063 | Compares two near-identical Applied AI pages; only one is retrieved. |
| cd054 | Chain of four links; the cap of 8 followed courses is used up by branches before ECE 202's entry is reached. |
| s013, s023, s019 | Partial answers: an expected fact is left out. |

Also seen, not fixed: for a course with no catalog entry the model can end with
"check the COSC 302 entry in the full catalog", which implies an entry exists.

## 6. Known limits of the judge

- It gets no page text, so it cannot check extra facts in an answer. It may under-count
  hallucinations, for either system.
- Judge and generator are both Claude models. Human grades are the headline on the test set.

## Spend

About $8.60 of API credit across all Phase 6 runs.
