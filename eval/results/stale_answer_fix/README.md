# Stale top-level answer after an update, 2026-11-28

**Bug.** After "my daughter meera has ballet on tuesdays 4pm" and, a day later, "update: ballet got rescheduled to
wednesdays", recall served the update first and the receipt said "wednesdays", but the top-level `answer` was
"tuesdays 4pm": `_best_answer` picked the most confident span among all accepted memories, and the update was only
recorded (`update_of`) when `promote_updates` had to move it. An update that already ranked first was never marked.

**Fix.** `promote_updates` also marks a later change-cue memory about the same subject that already ranks above the
earlier one (same tests, no reordering); `_best_answer` skips memories a later message updated, and gives no
top-level answer when only those hold one ("ballet is cancelled for good"). Updates without a cue word keep the
measured `later_answer` behaviour. Tests: tests/test_updates.py.

Development data only (LoCoMo dev, our dev set, Gemini batches 1-3); "before" = the last commit, same machine.

| | before | after |
|---|---|---|
| LoCoMo dev answer token-F1 | 0.3752 | 0.3758 |
| LoCoMo dev recall@10 | 0.7008 | 0.7008 |
| LoCoMo dev answerable wrongly declined | 0.0787 | 0.0787 |
| LoCoMo dev adversarial declined | 0.3249 | 0.3249 |
| LoCoMo dev time questions with the gold date | 68/156 | 68/156 |
| LoCoMo dev answers that changed | | 3 of 999 |
| Dev set first-answer match | 0.8571 | 0.9286 |
| Dev set stale top / false abstain | 0 / 0.0357 | 0 / 0.0357 |
| Gemini batches first-answer match | 0.8000 | 0.8000 |
| Gemini batches stale top / false abstain | 0 / 0.1200 | 0 / 0.1200 |

Commands: `eval/results/date_resolution/measure.py locomo10.json OUT` and
`eval/run_eval.py --split dev|fresh --embedder st:BAAI/bge-small-en-v1.5 --only retrieval --json OUT`.
