# Later answer for first-person "now" questions, 2026-11-28

**Question.** When recall finds two memories giving different answers to the question (recall.answer_conflicts),
the top-level `answer` keeps the reader's most confident one and reports the later one as `later_answer`. Real
assistant questions showed the cost: "my manager is priya" then "my new manager rohan seems nice so far" -> "Who is my
manager?" answered "priya"; same for "my dentist is dr mehta" -> "dr rao ... is my dentist going forward".

**LoCoMo dev (94 questions with a later, different answer; `measure_conflicts.py`):** the confident answer
matched the gold better on 52, the later one on 11 (31 ties); mean token-F1
0.454 vs 0.153. Even present-tense LoCoMo questions favour the confident answer
(13 vs 3): they are third-person questions about different facts ("What is John's main focus in local politics?"),
not updates. Switching every conflict to the later answer would lose many right answers.

**Change.** Only for a first-person question about the user's present ("Who is my manager?", "Where do I live?",
updates.asks_about_now: first person, present tense, no history words like was / did / originally / ever), the later
answer is the top-level answer. Everything else is unchanged.

**Result (development data only; "before" = the commit with the stale-answer fix, eval/results/stale_answer_fix):**
LoCoMo dev summary identical, 0 of 999 answers changed; our dev set and the Gemini batches identical (no question in
them has a detected conflict with a first-person "now" question). Targeted real-model checks: manager -> rohan,
dentist -> dr rao (both were the older answer); the ballet update still wednesdays.

**Not addressed.** Updates the conflict check does not notice at all (no shared topic words): "i live in delhi" ->
"found a flat in pune near my new office" -> "Where do I live?" still answers delhi; "i drive a honda city" ->
"sold the honda, bought a hyundai creta" still answers honda city. Detecting those without merging facts that are
both true ("places I've lived") needs a dedicated set of update / non-update cases to measure first.
