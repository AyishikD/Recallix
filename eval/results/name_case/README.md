# Capitalising names for the reader: not shipped, 2026-11-28

Question: the cased SQuAD reader answers "Which gym do I go to?" from "my gym is Cult Fit" (margin +3.7) but not from
"my gym is cult fit" (-10.9; sentence case alone, "My gym is cult fit.", -4.7, threshold -2.7). Does capitalising likely
names in all-lowercase memories before the reader help?

Experiment (code reverted): in all-lowercase text, capitalise words the system dictionary (/usr/share/dict/web2) does
not know as lowercase words, with simple suffix stripping and a short list of texting words. Three modes against the
current default (recall.answer_check.restore_case off): sentence case (the existing restore_case: true), names, and
names + sentence case. All data is development data (Gemini batches 1-3 have all been looked at).

| mode | Gemini answer_match | Gemini false declines | Gemini traps declined (of 30) | dev set | update set correct / wrong / declined |
|---|---|---|---|---|---|
| off (default) | 0.800 | 0.120 | 28 | 0.929 | 32 / 1 / 5 |
| sentence case | 0.867 | 0.080 | 26 | 0.929 | 32 / 1 / 5 |
| names | 0.827 | 0.107 | 27 | 0.929 | 30 / 2 / 6 |
| names + sentence | 0.853 | 0.067 | 25 | 0.929 | 32 / 2 / 3 |

Every mode answers more and also answers more trap questions wrongly; the name modes also add a wrong answer on the
update set (unrelated_mom_phone: "pixel 9", the user's new phone, for the mom's phone) and decline unrelated_brother_job.
Names alone is the worst trade (+2 answers on Gemini for +1 wrong trap and +1 wrong update answer). Under "a wrong answer
costs more than a refusal", none is shipped. The dictionary also is not present in the Linux image. The gym case is not
fixed by any mode ("cult" and "fit" are dictionary words).

Files: gemini_batches_*.json, dev_set_*.json, update_set_*.json (true = sentence case). Baseline files:
eval/results/update_set/*_after_attr.json.
