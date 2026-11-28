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

## Unseen Gemini batch 3 (confirmation)

eval/data/fresh/updates/gemini_upd3.jsonl (a resident doctor in Delhi), scored once with the code at 8cb3eeb3 before any
scenario was looked at; then with older code for comparison.

| code | correct | wrong | declined | other |
|---|---|---|---|---|
| before the day's update fixes (db109d48) | 14 | 6 | 8 | 2 |
| after the first fixes and the purpose rule (42e0c258) | 16 | 4 | 8 | 2 |
| after the batch 1-2 fixes (8cb3eeb3), first score | 16 | 4 | 8 | 2 |

Confirmed: the day's fixes up to 42e0c258 turn two wrong answers into correct ones (update_mumbai_flight "akasa air",
update_tiffin_service "annapurna tiffin service") with no new wrong answer. The batch 1-2 fixes (8cb3eeb3) did not change
any answer here: neither confirmed nor refuted.

All three unseen batches, first scores (90 scenarios): wrong 13 with the code before the fixes, 10 with the code at the
time of each score (4 -> 4, 3 -> 2, 6 -> 4).

Still wrong: update_ward_posting ("emergency medicine"; later "shifted to pediatric icu", which shares no word with
"Which department or ward am I currently posted in?"), update_stethoscope, update_air_purifier (both "What ... did I
order / buy?": past-tense questions, where the latest purchase is the answer), update_scrubs_color ("navy blue").
Declines are high in the unrelated group (5 of 9).

Files: gemini_upd3.json (first score), gemini_upd3_before_fixes.json, gemini_upd3_at_42e0c258.json.

## LLM reference (gpt-oss-120b on Groq, 2026-11-28)

`eval/llm_ceiling.py`: the model reads all of a scenario's messages (in order, with day numbers) and the question, and
answers with a short phrase from the messages or "unknown"; same scoring. One prompt, temperature 0, reasoning effort
low, 29,043 tokens for the 90 scenarios of gemini_upd1-3. File: llm_ceiling_gpt-oss-120b.json.

| | update (36) correct / wrong / declined / other | both_true (27) | unrelated (27) | all (90) |
|---|---|---|---|---|
| LLM | 26 / 8 / 0 / 2 | 27 / 0 / 0 / 0 | 23 / 0 / 0 / 4 | 76 / 8 / 0 / 6 |
| Recallic, first scores (unseen) | 18 / 8 / 6 / 4 | 21 / 1 / 3 / 2 | 15 / 1 / 6 / 5 | 54 / 10 / 15 / 11 |
| Recallic, current code (tuned on batches 1-2) | 20 / 5 / 7 / 4 | 22 / 0 / 3 / 2 | 15 / 0 / 6 / 6 | 57 / 5 / 16 / 12 |

Read by hand, the LLM's "other" answers are mostly right in a different form ("302", "16000", "-2.5", "carpool"); one is
wrong ("acer" for the car service). Findings:
* Where the LLM is far ahead is answering at all: it never declines, and it is right on every both_true and unrelated
  scenario, where Recallic declines 9 of 54 and gives partial spans.
* On updates the LLM is wrong as often as Recallic was unseen (8 of 36 vs 8): it answered the outdated value for the milk
  brand, the school-bus stop, both flights, the desk, the consulting fee, the spectacles and the duty day. Recallic
  (current code) gets four of those right (bus stop, consulting fee, Mumbai flight, car service).
* For a learned update model, the labels should come from how the scenarios are built (the generator knows which
  message replaces which), not from an LLM answering them: on updates the LLM is right only about 3 times in 4.
