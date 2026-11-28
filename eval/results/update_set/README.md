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
