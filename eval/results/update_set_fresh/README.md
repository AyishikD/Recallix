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
