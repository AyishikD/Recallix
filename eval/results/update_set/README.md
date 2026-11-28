# Update set: updates without cue words, 2026-11-28

Development data written for this question (eval/data/dev/updates/scenarios.jsonl, 39 first-person scenarios, a few
messages a day apart each). Scored on the top-level `answer` by `eval/updates_eval.py`; correct = an expected value and
no forbidden one, wrong = a forbidden value (the outdated fact, or someone else's).

| group | n | correct | wrong | declined | other |
|---|---|---|---|---|---|
| update: the later message replaces the fact | 14 | 6 | 8 | 0 | 0 |
| both true: a later fact must not replace it | 13 | 12 | 0 | 0 | 1 |
| unrelated: someone else's later fact | 12 | 9 | 2 | 1 | 0 |

Baseline = the code after eval/results/later_answer (db109d48). Findings:
* Most cue-less updates are missed: the earlier fact is answered (home, car, phone, employer, laptop, bank) because the
  two messages share no topic words, so the conflict check never pairs them.
* In two update cases the first served memory already holds the new value (diet: vegan, school: inventure) but the
  top-level answer is the older, more confident one (no conflict marked).
* A later fact about someone else answered for the user in two cases ("my sister's manager is rohan" -> "Who is my
  manager?" -> rohan; a friend's gym): an attribution problem, separate from updates.
* The both-true group is already handled (12/13; the 13th is a reader span: "peanuts" for "which medicine").

## After the fixes (same day)

| group | correct | wrong | declined | other |
|---|---|---|---|---|
| update | 6 -> **10** | 8 -> **1** | 0 -> 3 | 0 -> 0 |
| both true | 12 -> 12 | 0 -> 0 | 0 -> 0 | 1 -> 1 |
| unrelated | 9 -> 10 | 2 -> **1** | 1 -> 1 | 0 -> 0 |

Changes (backend/brain/memory/updates.py, recall_engine.py `_best_answer`, all only for first-person questions about now,
`asks_about_now`, except the possessive fix):
1. **Latest answer wins** when answers differ, among answers stated about the user (`about_the_user`: I/my/me/we, or a
   texting sentence starting with a verb) whose margin is at least `recall.answer_conflicts.override_min_margin` (-0.5).
2. **Ended facts**: a value a later message says ended ("resigned from infosys", "closed the hdfc account", "sold the
   honda", "the iphone went to my mom") is not answered; no answer rather than the outdated one.
3. **Possessives** in the other-people filter: "my sister's manager is rohan" is the sister's, not the user's.

The guard values were set on development data: the dev set's "Where do I work?" (a later "I moved to Mumbai", margin
-1.10) and "Where do I live now?" (a later "Anita lives in Bangalore") regressed without them.

Gates (development data): LoCoMo dev identical (0 of 999 answers changed); our dev set and the Gemini batches
identical to the previous commit. Files: after.json, locomo_dev_after.json, dev_set_after.json, gemini_batches_after.json.

Still wrong: "Which school does Meera go to?" (a named person's present: third person is left to the confident answer,
which LoCoMo favours) and a friend's gym mentioned by first name only ("rahul keeps telling me golds gym is better, he
goes there"). Phone, employer and bank now decline instead of answering the outdated value; the reader does not pick the
new value from those messages.

## Attribution: someone telling the user (same day)

`without_other_people` now also sets aside a clause where another person tells, says, recommends, suggests or insists
something to the user ("rahul keeps telling me golds gym is better, he goes there"); a following he/she clause continues
it. Kept when the question names that person ("Which gym does Rahul recommend?") and when the clause is advice to the
user ("the doctor told me to take vitamin d", "asked me not to ...").

| group | correct | wrong | declined | other |
|---|---|---|---|---|
| unrelated | 10 -> 10 | 1 -> **0** | 1 -> 2 | 0 -> 0 |

Update set: 1 wrong left (of 39), from 2; no other scenario changed. The friend's gym now declines rather than answering
"cult fit": the reader does not answer "Which gym do I go to?" from the lowercase "my gym is cult fit" even alone
(margin -10.9; "My gym is Cult Fit" and "What is my gym?" answer), a reader limit separate from attribution.

Gates (development data): LoCoMo dev identical (0 of 999 answers changed); dev set and Gemini batches identical apart
from timings. Files: after_attr.json, locomo_dev_after_attr.json, dev_set_after_attr.json, gemini_batches_after_attr.json.

## A stated change for someone else (same day)

For a present question about someone else ("Which school does Meera go to?"), the later conflicting answer now replaces
the more confident earlier one only when its memory says something changed (shifted, switched, transferred, moved,
changed, instead) and both memories name the same word of the question (`states_a_change` in updates.py, used in
recall_engine.py `_best_answer`). "shifted meera to inventure academy this term" (margin -0.92) now answers over "meera
goes to dps whitefield" (8.23). Without a change word ("arjun goes to inventure academy") or without the person ("tara
moved to inventure academy") the most confident answer stays.

| group | correct | wrong | declined | other |
|---|---|---|---|---|
| update | 10 -> **11** | 1 -> **0** | 3 -> 3 | 0 -> 0 |

Update set: 0 wrong of 39 (from 1); no other scenario changed. Gates (development data): LoCoMo dev identical (0 of 999
answers changed: the rule did not fire there); dev set and Gemini batches identical. Files: after_change.json,
locomo_dev_after_change.json, dev_set_after_change.json, gemini_batches_after_change.json.
