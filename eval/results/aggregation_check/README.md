# Aggregation check, 2026-11-26

Counting, ordering and time-between questions ("How many weddings did I attend?", "Which came first, A or
B?", "How many days after X did Y happen?") have no answer phrase inside one memory. The extractive answer
check could not judge them: it declined real questions and answered traps at nearly the same rate, and
"how many times" questions skipped it altogether (`recall.answer_check.calculation_questions`).

`recall.aggregation_check` (backend/brain/features/aggregation_check.py) decides these questions differently:
the question is rewritten by rules into the events it takes for granted ("I did attend weddings.", "My
knitting class happened."), and an entailment model (cross-encoder/nli-deberta-v3-base, Apache 2.0, local,
no LLM) checks that some ranked memory supports each one. All supported: serve. One unsupported: decline.
Questions the rules do not recognise are untouched. Off by default.

Development data only. No LLM, no API cost.

## Aggregation dev set (eval/data/dev/aggregation, 51 scenarios, 183 answerable questions, 56 traps)

`python eval/aggregation.py run --embedder st:BAAI/bge-small-en-v1.5 [--config <aggregation_check on>]`
Files: `aggregation_dev_before.json`, `aggregation_dev_after.json`.

| | answerable declined | all evidence served | traps declined | median recall |
|---|---|---|---|---|
| Before | 57/183 (0.311) | 0.661 | 17/56 (0.304) | 65 ms |
| After | 7/183 (0.038) | 0.907 | 50/56 (0.893) | 128 ms |

| Category | answerable declined | all evidence served | traps declined |
|---|---|---|---|
| Counting (60 / 22 traps) | 0.267 → 0.033 | 0.650 → 0.817 | 0.273 → 0.864 |
| Time-between (56 / 16) | 0.232 → 0.089 | 0.768 → 0.911 | 0.312 → 0.812 |
| Ordering (67 / 18) | 0.418 → 0.000 | 0.582 → 0.985 | 0.333 → 1.000 |

The rewriting rules were written while looking at this set, so these numbers are an upper bound, not a
held-out result.

## LoCoMo dev (conversations 1-5, 999 questions): the no-harm check

The rules never saw LoCoMo's wording before the first run. `retrieval_fixes/measure.py --split dev`.
Files: `locomo_dev_before.json`, `locomo_dev_after.json`.

| | served recall@10 | answerable wrongly declined | traps declined | median recall |
|---|---|---|---|---|
| Before | 0.7008 | 60/760 | 77/237 | 92 ms |
| After | 0.6996 | 59/760 | 77/237 | 100 ms |

15 of 999 questions are recognised as aggregation questions; 7 change: 4 that were declined are now
answered, 3 that were answered are now declined (their wording is looser than the memory: "adopting Coco"
against "I got a puppy ... her name's Coco"). LoCoMo's traps are not aggregation questions, so it cannot
show the trap gain; it shows the check does no harm elsewhere. A recognised question takes about 430 ms
instead of 115 ms here (ten long messages read by the entailment model); other questions are unchanged.

The first LoCoMo dev run, with rules that only handled "I", wrongly declined 62/760: questions about named
speakers ("How many times has John injured his ankle?") were rewritten badly. The rules now keep the
question's own subject, and a statement about a memory's speaker is also read in the first person.

## What it still gets wrong (aggregation dev)

- Traps answered (6): the entailment model accepts a wedding as "a mundan ceremony" and a filling as "a
  root canal"; a booked gig that was later cancelled counts as performed; two traps hinge on a date or on
  "I mentioned" being literally true.
- Real questions declined (7): the memory needs world knowledge the model lacks ("tried Ramiro tonight"
  as trying a restaurant in Lisbon), or the event's memory is not in the top 10.

## Not yet done

Held-out confirmation. LongMemEval is to be run once, after the preference category is added.
