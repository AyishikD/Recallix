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
| After | 7/183 (0.038) | 0.907 | 50/56 (0.893) | 129 ms |

| Category | answerable declined | all evidence served | traps declined |
|---|---|---|---|
| Counting (60 / 22 traps) | 0.267 → 0.033 | 0.650 → 0.817 | 0.273 → 0.864 |
| Time-between (56 / 16) | 0.232 → 0.089 | 0.768 → 0.911 | 0.312 → 0.812 |
| Ordering (67 / 18) | 0.418 → 0.000 | 0.582 → 0.985 | 0.333 → 1.000 |

The rewriting rules were written while looking at this set, so these numbers are an upper bound, not a
held-out result. The next section is the fairer test.

## Reworded questions (eval/data/dev/aggregation/paraphrases.jsonl)

Two rewordings of each of the 239 questions (478), written by a separate agent that saw only
scenarios.jsonl, never the code or the rules ("Baby showers - how many have I actually attended?",
"What was the order in which I sent A and B?", "I signed the lease - how many days later did I move in?").
The answer keys are unchanged. Scenarios at even positions are half A, at odd positions half B. Half A's
wording was read while extending the rules; half B's wording was never read, only its totals.
`python eval/aggregation.py run --paraphrases eval/data/dev/aggregation/paraphrases.jsonl --half A|B`.
Files: `paraphrases/{A,B}_{before,first_rules,after}.json`.

| | answerable declined | all evidence served | traps declined |
|---|---|---|---|
| Half A, check off | 67/188 (0.356) | 0.606 | 21/56 (0.375) |
| Half A, first rules | 53/188 (0.282) | 0.676 | 33/56 (0.589) |
| Half A, rules now | 8/188 (0.043) | 0.899 | 48/56 (0.857) |
| **Half B (wording unseen), check off** | 50/178 (0.281) | 0.691 | 14/56 (0.250) |
| Half B, first rules | 49/178 (0.275) | 0.702 | 28/56 (0.500) |
| **Half B, rules now** | 20/178 (0.112) | 0.865 | 45/56 (0.804) |

The first rules (textbook forms only: "How many X did I ...", "Which came first, A or B") recognised 84 of
244 rewordings in half A and 76 of 234 in half B; everything else fell back to the old behaviour. The rules
now also read a fronted topic, counts asked without "how many", and events stated before the question;
they recognise 244 of 244 in A and 220 of 234 in B. Half B is the number to quote: its wording was not
used. It shares scenarios' style and author with half A, so it is still development data.

By category on half B: counting 0.138 declined / 0.909 traps declined; time-between 0.167 / 0.625;
ordering 0.045 / 0.833.

## LoCoMo dev (conversations 1-5, 999 questions): the no-harm check

The rules never saw LoCoMo's wording before the first run. `retrieval_fixes/measure.py --split dev`.
Files: `locomo_dev_before.json`, `locomo_dev_after.json`.

| | served recall@10 | answerable wrongly declined | traps declined | median recall |
|---|---|---|---|---|
| Before | 0.7008 | 60/760 | 77/237 | 92 ms |
| After | 0.6996 | 59/760 | 77/237 | 92 ms |

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

## Tried and rejected

- Declining when a memory contradicts a presupposed event (meant for "booked the gig", then "the gig got
  cancelled"): the entailment model marks unrelated memories as contradictions. Offline on the dev set:
  answerable questions answered 176 -> 64 of 183 for 4 more traps declined.
- Also reading the memories `recall.second_pass` would add before declining: no decision changed on either set.
- cross-encoder/nli-deberta-v3-xsmall instead of -base: offline 134 vs 163 of 183 answerable answered.
- Treating "How often does John ...?" as a count: it asks for a habit that one memory states; on LoCoMo dev
  it turned a declined trap into an answer. Only past-tense "how often did/have ..." counts.

## Not yet done

Held-out confirmation. LongMemEval is to be run once, after the preference category is added.
