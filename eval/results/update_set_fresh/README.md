# Update scenarios, unseen Gemini batch 1 (2026-11-28)

eval/data/fresh/updates/gemini_upd1.jsonl, 30 scenarios, scored once before any of them was looked at, with the code
at da889629 (after the update-set fixes of the same day). For comparison, the same batch with the code before those
fixes (db109d48), run afterwards.

| code | correct | wrong | declined | other |
|---|---|---|---|---|
| before the update fixes (db109d48) | 17 | 4 | 5 | 4 |
| after (da889629), first score | 17 | 4 | 5 | 4 |

By group (after): update 6 / 2 / 2 / 2, both_true 5 / 1 / 2 / 1, unrelated 6 / 1 / 1 / 1 (correct / wrong / declined /
other). On our own update set the same fixes went from 10 wrong to 0 of 39: much of that did not carry over.

Changed by the fixes: update_bus_stop wrong -> correct ("my home bhel gate"); both_true_credit_card correct -> wrong
("Which credit card do I use for booking flights?" now answers the later "sbi cashback card": the first-person
latest-wins rule replaced a fact that was still true). Net: same number of wrong answers.

Still wrong (both versions): update_commute ("metro" for the later quick ride carpool), update_cook_salary ("7000 rs"
for the later 8500), unrelated_blood_group ("b negative", the wife's, for "What is my blood group?").

Files: gemini_upd1.json (first score), gemini_upd1_before_fixes.json.

## Fix: a purpose in the question (same day, development data from here on)

After looking at the failures (gemini_upd1 is development data from now on). "Which credit card do I use for booking
flights?": the first-person latest-wins rule let the later "sbi cashback card ... for swiggy and amazon orders" (margin
-0.14) replace "hdfc regalia for all flight bookings" (2.64). Now, when the question narrows the topic with "for ..."
(`purpose_stems`, 4-letter stems of the words after "for", time words left out) and some competing memories match it,
only those compete for latest-wins (`matches_purpose`; recall_engine.py `_best_answer`). When none match, nothing changes.

| set | correct | wrong | declined | other |
|---|---|---|---|---|
| gemini_upd1 | 17 -> 18 | 4 -> **3** | 5 -> 5 | 4 -> 4 |
| our update set (eval/data/dev/updates) | 33 -> 33 | 0 -> 0 | 5 -> 5 | 1 -> 1 |

Only both_true_credit_card changed. Gates: LoCoMo dev identical (0 of 999 answers changed); dev set and Gemini batches
identical. To be confirmed on a later unseen batch. Files: gemini_upd1_after_purpose.json, locomo_dev_after_purpose.json,
dev_set_after_purpose.json, gemini_batches_after_purpose.json; our update set: eval/results/update_set/after_purpose.json.

## Unseen Gemini batch 2 (confirmation)

eval/data/fresh/updates/gemini_upd2.jsonl (another persona), scored once with the code at 42e0c258 before any scenario
was looked at; the code before the day's update fixes (db109d48) run afterwards for comparison.

| code | correct | wrong | declined | other |
|---|---|---|---|---|
| before the update fixes (db109d48) | 20 | 3 | 2 | 5 |
| after (42e0c258), first score | **21** | **2** | 2 | 5 |

Changed: update_temple_day wrong ("tuesday") -> correct ("friday"); nothing else. No new wrong answer. Over both unseen
batches the day's fixes (as first scored) went from 7 to 6 wrong of 60, plus the credit-card fix tuned on batch 1.

Read by hand, "other" includes one answer that is really wrong: unrelated_flat_number answers "104", the chairman's flat,
for "What is my flat number?" (the scorer needs "flat 302"). The rest are partial spans of the right value ("80" for
telma 80, "silver wagon", "prabhat", "maharashtra bajirao road branch").

Still wrong: update_society_agm ("15th september"; postponed to the 29th), update_security_agency ("eagle security";
terminated, tiger force hired).

Files: gemini_upd2.json (first score), gemini_upd2_before_fixes.json.

## Fixes after both batches (same day; both batches are development data now)

Changes (updates.py, recall_engine.py `_best_answer`):
1. "we", "our" and "us" make a question first person ("How much do we pay the cook?", "When is our society AGM?").
2. A first-person question is not answered from a memory that gives the asked thing as someone else's ("her blood group
   is b negative" for "What is my blood group?", `someone_elses`).
3. "terminated" ends a value; a bare "gave up" ends what the clause before it named ("metro ... parking fee increased,
   gave up and booked quick ride carpool": the metro is no longer the answer).
4. A memory that states a change (shifted, switched, moved, postponed, rescheduled, revised, instead ...) and shares a
   word with the question may replace the earlier answer without the margin floor ("postponing agm to 29th september").

| set | correct | wrong | declined | other |
|---|---|---|---|---|
| gemini_upd1 | 18 -> 19 | 3 -> **0** | 5 -> 6 | 4 -> 5 |
| gemini_upd2 | 21 -> 22 | 2 -> **1** | 2 -> 2 | 5 -> 5 |
| our update set | 33 -> 33 | 0 -> 0 | 5 -> 5 | 1 -> 1 |

Changed: update_cook_salary -> "8500 rs"; update_society_agm -> "29th september morning"; update_commute wrong -> declined;
unrelated_blood_group "b negative" -> "positive" (the user's value, cut short). Still wrong: update_security_agency ("Which
security agency manages our society guards?" is not read as a present-tense question: "manages" is not in the
present-tense words). Still wrong by hand: unrelated_flat_number ("104").

Gates: LoCoMo dev identical (0 of 999 answers changed); dev set and Gemini retrieval batches identical. These fixes were
made after looking at both batches and need a third unseen batch to be confirmed. Files: upd1_after_ended.json,
upd2_after_ended.json, dev_after_ended.json, locomo_dev_after_ended.json, dev_set_after_ended.json,
gemini_batches_after_ended.json.
