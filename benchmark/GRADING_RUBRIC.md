# Grading Rubric

Written 2026-10-30, before any system output or baseline response was collected.
This file is frozen. If a rule turns out to be ambiguous during grading, the
clarification is appended under "Clarification log" with a date, and never
changes a grade already given.

The same rubric is used by the human grader and by the LLM judge. The judge
receives this file verbatim as its instructions.

## What the grader sees

For each question: the question text, the expected answer and source pages
from the question set, and one response labeled "Response A" or "Response B".
The grader does not know which system produced the response.

## Step 1: answer label (exactly one)

Decide in this order. The first rule that applies wins.

| Label | Definition |
|---|---|
| `correct_refusal` | The question is in the `unanswerable` category and the response says the catalog does not contain the answer (or that it cannot answer from the catalog), without asserting a specific answer. Pointing the student elsewhere ("ask your advisor", "check the timetable") is fine. |
| `wrong_refusal` | The question is answerable (any category except `unanswerable`) and the response declines to answer, says the information is not available, or only redirects elsewhere, without giving the expected answer. |
| `hallucinated` | The response states, with confidence, at least one specific fact about UTK courses or programs that is false according to the catalog (wrong credit hours, invented prerequisite, nonexistent course, wrong requirement). This label applies even if other parts of the answer are right, and applies to unanswerable questions when the response invents an answer. Hedged speculation clearly marked as uncertain ("I think", "typically") is not hallucinated; grade it under the rules below. |
| `correct` | Every fact in the expected answer is present and accurate. Extra accurate information does not hurt. Different wording, ordering, or formatting does not hurt. For list answers (e.g. prerequisites), all required items are present and no required item is missing. |
| `partially_correct` | Some, but not all, facts in the expected answer are present and accurate, and nothing stated is false. Typical cases: lists one of two prerequisites; gives credit hours but omits the corequisite the question asked for; answers one half of a comparison. |
| `incorrect` | The response attempts an answer, states nothing demonstrably false with confidence, but does not contain the expected facts (e.g. answers a different question, describes the course without giving the asked fact, or is too vague to contain the answer). |

Edge rules:

- A response that gives the expected answer and then adds a false fact is `hallucinated`, not `correct`. Hallucination is graded on the whole response.
- A response that refuses and then speculates ("the catalog doesn't say, but it's probably 3 hours") is graded on the speculation: `hallucinated` if false, `partially_correct`/`correct` if true, since the student receives the guess either way.
- Information that is true but comes from outside the catalog (a department web page, the timetable) counts as accurate if it does not contradict the catalog. The benchmark measures correctness, not provenance; provenance is scored in Step 2.
- Course codes must match. "COSC 302" and "COSC302" and "cosc 302" are the same course. "COSC 302" and "COSC 312" are not.
- Expected answers are short. Do not require the response to repeat the exact phrasing of the catalog.
- If the grader cannot decide between two labels after reading the source page, choose the lower one in this order: `correct` > `partially_correct` > `incorrect`, and record the doubt in the notes column.

## Step 2: citation scores (two yes/no fields)

Graded in a separate pass when citations cannot be hidden, because citation style can reveal the system.

| Field | Rule |
|---|---|
| `cited` | `yes` if the response names a source: a URL, a page title, or an unambiguous reference such as "COSC 302 course description page" or "the Computer Science BS program page". "According to the catalog" with no page is `no`. |
| `citation_correct` | `yes` if at least one cited source is a page that actually contains the expected answer (one of the question's `source_urls`, or a page whose title matches one of them). `no` if cited sources do not contain the answer or do not exist. `na` when `cited` is `no`. |

For `correct_refusal` responses, `citation_correct` is `na` (there is no answer to point at).

## Step 3: notes

Free text. Required when a grade is `hallucinated` (quote the false claim) or when the grader used the tie-break rule.

## Metrics derived from these labels (defined now, not after results)

- **Accuracy** = (`correct` + `correct_refusal`) / all questions. This is the headline number. `partially_correct` does not count toward accuracy.
- **Partial-credit accuracy** = (`correct` + `correct_refusal` + 0.5 × `partially_correct`) / all questions. Reported as a secondary number.
- **Hallucination rate** = `hallucinated` / all questions.
- **Refusal precision** = `correct_refusal` / (`correct_refusal` + `wrong_refusal`).
- **Citation rate** = `cited == yes` / responses that attempted an answer (not refusals).
- **Citation accuracy** = `citation_correct == yes` / `cited == yes`.
- **Human-judge agreement** = exact label agreement rate plus Cohen's kappa over the six labels, reported for the test set.

## Examples

Question: "How many credit hours is COSC 311?" Expected: "3".

- "COSC 311 Discrete Structures is 3 credit hours." → `correct`
- "COSC 311 is a 3-hour course that requires COSC 202 and MATH 142 or MATH 148." → `correct` (extra facts are true)
- "COSC 311 is 4 credit hours." → `hallucinated`
- "COSC 311 covers sets, functions, and proof techniques." → `incorrect`
- "I don't have information about credit hours." → `wrong_refusal`

Question: "What are the prerequisites for COSC 311?" Expected: "COSC 202 with a C or better, and MATH 142 or MATH 148 with a C or better."

- "COSC 202 and either MATH 142 or MATH 148, each with a grade of C or better." → `correct`
- "COSC 202." → `partially_correct`
- "COSC 202 and MATH 141." → `hallucinated` (MATH 141 is not a listed prerequisite)

Question (unanswerable): "Who teaches COSC 311 in Spring 2027?"

- "The catalog does not list instructors. Check the timetable or the department." → `correct_refusal`
- "COSC 311 is taught by Dr. Smith." → `hallucinated`
- "The catalog doesn't say, but it's usually Dr. Smith." → `hallucinated` (speculated name)

## Clarification log

(empty)
