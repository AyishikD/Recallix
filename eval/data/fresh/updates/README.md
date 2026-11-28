# Fresh update scenarios (independently written)

Written by Google Gemini from a task description only (no rules, test sets or known gaps), prompted by the project
owner with one persona (a 34-year-old product manager in Hyderabad). Same format and scoring as
eval/data/dev/updates (eval/updates_eval.py --data ...). Messages, questions and answer keys unedited.

| Batch | Date | Scenarios | Answer-key check | SHA-256 |
|---|---|---|---|---|
| gemini_upd1 | 2026-11-28 | 30 (12 update, 9 both_true, 9 unrelated) | every value appears verbatim in a message; no expected value overlaps a forbidden one | `e3e0c70d16c5102d88d8c6db817e5432d73aa04a496ab4f2014cc23344096485` |
| gemini_upd2 | 2026-11-28 | 30 (12 update, 9 both_true, 9 unrelated); persona: a 58-year-old retired bank officer in Pune | same checks, all pass | `bf21afeabea21fe6891410f9f64d0efba2f344c4928d16e62bbe3d8f11481db1` |

Protocol: a batch is scored once with the frozen code; after its failures have been looked at it counts as development
data, and fixes motivated by it must be confirmed on a later batch. gemini_upd1 and gemini_upd2 are now development data
(looked at 2026-11-28).
