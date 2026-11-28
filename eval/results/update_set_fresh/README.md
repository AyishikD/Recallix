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
