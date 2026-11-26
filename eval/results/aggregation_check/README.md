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
| **Half B, rules now** | 16/178 (0.090) | 0.888 | 42/56 (0.750) |

The first rules (textbook forms only: "How many X did I ...", "Which came first, A or B") recognised 84 of
244 rewordings in half A and 76 of 234 in half B; everything else fell back to the old behaviour. The rules
now also read a fronted topic, counts asked without "how many", and events stated before the question;
they recognise 244 of 244 in A and 220 of 234 in B. Half B is the number to quote: its wording was not
used. It shares scenarios' style and author with half A, so it is still development data.

By category on half B (answerable declined / traps declined): counting 0.138 / 0.773; time-between 0.093 / 0.625; ordering 0.045 / 0.833.
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

## Hard set from a different author (eval/data/dev/aggregation/hard_scenarios.jsonl)

12 scenarios (47 answerable questions, 24 traps) written by a different AI model from a task description
only, with deliberate distractors in every scenario: a cancelled plan, someone else's event of the same
kind, a similar-but-different event, a follow-up message, and events worded differently in the question
and the message. A separate agent audited the answer keys (hard_review.jsonl: 3 accepted, 9 fixed; all four
ordering scenarios had a "trap" that was answerable). Recallix was then measured ONCE before anyone read the
questions (`hard/before.json`, `hard/after_first_run.json`); after that the failures were read and the
rules extended, so the last row is development data.

| | answerable declined | all evidence served | traps declined |
|---|---|---|---|
| Check off | 17/47 | 0.617 | 11/24 |
| **Check on, first run (questions unseen)** | 17/47 | 0.617 | 19/24 |
| Check on, after fixing what it showed | 7/47 | 0.830 | 19/24 |

Unseen, the check kept its trap protection and gave no net gain on answerable questions (ordering better,
time-between worse). What it showed: the rewriting assumed the person did every event themselves ("the
movers loading my container" became "I was loading the container", which the model rightly rejects), and
messy phrasings garbled the statement ("Count the separate occasions on which ...", a chatty lead-in before
a dash, "Did I have X before or after Y?"). Fixes: a gerund event is also read without a doer ("Loading
the container happened"), and those phrasings are read. The same change moved half B of the reworded
questions from 20 to 16 answerable declined and from 45 to 42 traps declined: the doer-less reading
lets a few more traps through. The rules recognise 62 of 71 hard-set questions.

## Second hard batch, measured once and not read (hard2_scenarios.jsonl)

Same prompt and author as the hard set, new themes; 12 scenarios, 47 answerable questions, 24 traps.
Keys audited by a separate agent (hard2_review.jsonl: 3 accepted, 9 fixed; again four ordering "traps" were
answerable). Measured once, with the rules exactly as they were after the first hard set. Nobody working
on the rules has read its questions or its per-question results; it is kept that way as a check set.
Files: `hard2/before.json`, `hard2/after_first_run.json`.

| | answerable declined | all evidence served | traps declined |
|---|---|---|---|
| Check off | 15/47 | 0.681 | 11/24 |
| **Check on (first and only run)** | 8/47 | 0.830 | 15/24 |

By category (answerable declined, traps declined): counting 1/15 -> 3/15, 3/8 -> 7/8; time-between
6/16 -> 3/16, 4/8 -> 4/8; ordering 8/16 -> 2/16, 4/8 -> 4/8. The rules recognised 58 of 71 questions.
All evidence was in the top 10 for every answerable question, with the check on or off.

## Plans are not events (plan rule)

Found while building the investor deck: on six messages ("booked flights to goa for march!!", later
"ugh goa is off. cancelled everything"), "How many times did I fly to Goa?" was answered, because the
entailment model reads a booking as the trip. Now a memory that only records a plan (booked, reserved,
registered, signed up, scheduled, planning, tickets for; or "will", "going to", "my next" with a named future
time) cannot support a claim that the event happened, unless the claim is itself about the plan ("How many
days after booking the flights did I cancel them?"). A first version also treated "can't wait" and "my
next" alone as plans; reading the lost questions on the readable sets showed that was too broad, and "-ing"
events now also get a "did" reading ("I did sign the paperwork"). Files: `plan_rule/`.

| Set | answerable declined | traps declined |
|---|---|---|
| Dev originals | 7 -> 9 of 183 | 50 -> 53 of 56 |
| Reworded A | 8 -> 8 of 188 | 48 -> 48 of 56 |
| Hard set | 7 -> 7 of 47 | 19 -> 19 of 24 |
| Reworded B (totals only) | 16 -> 20 of 178 | 42 -> 47 of 56 |
| Hard2, unread (totals only) | 8 -> 8 of 47 | 15 -> 16 of 24 |
| All five | 46 -> 52 | 174 -> 183 |

Kept: 9 more traps declined for 6 more refusals, in line with the rule that a wrong answer costs more than
a refusal. The receipt names the plan it set aside ("Memory 1 only records a plan for it, not the event").
Known loss: "Signed up for a language exchange app" is both the plan and the event.

## LongMemEval_S, held out for this check (27 November 2026): it does not confirm

LongMemEval had been run before (see eval/README.md), but never with this check, and no rule here was
written from its questions. Both passes use the same stored data (`.phase6_data/lme_store`, rebuilt for this
run) and today's code; the second pass has `recall.aggregation_check` on (with the plan rule). Files:
`longmemeval/before.json`, `after.json`, `rows_before.json`, `rows_after.json`.

| | check off | check on |
|---|---|---|
| Answerable wrongly declined (of 470) | 78 (0.166) | 89 (0.189) |
| Unanswerable declined (of 30) | 16 (0.533) | 19 (0.633) |

| Question type | served recall@10 | hit@10 |
|---|---|---|
| overall | 0.726 -> 0.701 | 0.789 -> 0.764 |
| temporal-reasoning | 0.622 -> 0.561 | 0.685 -> 0.606 |
| multi-session | 0.561 -> 0.577 | 0.703 -> 0.736 |
| knowledge-update | 0.928 -> 0.859 | 0.972 -> 0.903 |
| single-session-user | 0.953 -> 0.938 | 0.953 -> 0.938 |
| single-session-assistant | 0.786 -> 0.786 | 0.786 -> 0.786 |
| single-session-preference | 0.744 -> 0.744 | 0.800 -> 0.800 |

The rules recognised 148 of the 500 questions (136 answerable, 12 unanswerable). Among them, 23 answerable
questions that used to be declined are now answered and 34 that used to be answered are now declined (32
of those 34 had the evidence in the top 10), and 2 more unanswerable ones are declined. Net: more traps
declined (16 -> 19 of 30 overall) but more real questions refused, and lower served recall, mainly on
temporal-reasoning and knowledge-update questions. Even weighting a wrong answer as three refusals
(`learning.answer_check.wrong_answer_cost`), 3 traps x 3 = 9 does not cover 11 extra refusals.

**Decision: the check stays off by default.** Its gains on our own sets did not carry over to LongMemEval's
questions. LongMemEval has now been used for this check, so any change made after looking at why must be
confirmed on other fresh data (a new hard batch, or LongMemEval_M), not on LongMemEval_S again.

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
