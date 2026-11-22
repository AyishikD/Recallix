# Fresh test data (independently written)

Written by Google Gemini from a task description only. It never saw Recallix's rules,
test sets or known gaps. Prompted by the project owner with one persona per batch.
Messages and questions are **unedited**. Only a `name` per scenario was added.

| Batch | Date | Scenarios | Questions | Answer-key check |
|---|---|---|---|---|
| gemini1 | 2026-11-22 | 5 | 35 (25 answerable, 10 not) | every `expect` appears verbatim in a message; "frankfurt" and "blume" appear in two messages each, so either counts |
| gemini2 | 2026-11-22 | 5 | 35 (25 answerable, 10 not) | every `expect` appears verbatim; "waffle" and "dr harris" appear in two messages each (both correct) |
| gemini3 | 2026-11-22 | 5 | 35 (25 answerable, 10 not) | every `expect` appears verbatim in exactly one message |

retrieval.jsonl SHA-256 (batches 1-3): `5abd0cdf5964e84d9c1191710a7115da4bcec5dcc140cef283fccc1be69397a8`; batches 1-2 (first 10 lines): `a0d3ffe94b594f9be2c0e3f41e7da1d1c389e432d51c21f6c26c303cbe7d034e`; batch 1 alone (first 5 lines) was `45a7677e06c0c1ba9f47da1c2123e8b6d5e4c6b355c4a6414726236a78436437`

**Protocol:** each batch is scored once with the frozen configuration
(`python eval/run_eval.py --split fresh --only retrieval --scenarios gemini2_`), and the result is kept in
eval/results/. Once a batch's failures have been looked at, the batch counts as
development data. Fixes motivated by it must be confirmed on a *later* batch.

## Scores per batch

| Batch | When scored | Status | hit@1 | false declines | traps declined | File |
|---|---|---|---|---|---|---|
| gemini1 | first look | unseen | 0.76 | 0.08 | 9/10 | (before the results file existed; see the memory log) |
| gemini2 | confirmation of updates + user_perspective | unseen | 0.80 (same with both off) | 0.16 | 10/10 | `fresh_gemini2_all-MiniLM-L6-v2.json`, `..._changes_off.json` |
| gemini1+2 | after reader case restoration and the quantity/purchase update cues | **development** (tuned on) | 0.92 | 0.08 | 18/20 | `fresh_dev_batches1-2_all-MiniLM-L6-v2.json` |
| gemini3 | confirmation of case restoration + quantity/purchase cues (frozen) | unseen | 0.80 | 0.04 | 8/10 | `fresh_gemini3_all-MiniLM-L6-v2.json` |
| gemini3 | same, both of those off | unseen | 0.80 | 0.08 | 10/10 | `..._before_reader_fix.json` |
| gemini3 | everything off (also updates, user_perspective) | unseen | 0.80 | 0.08 | 10/10 | `..._all_off.json` |

Gemini batch 3: case restoration answered 1 more question (and declined 1 fewer answerable one), but
let 2 trap questions through ("organic chemistry" as the user's major, when it was the roommate's; a
$45 entry fee as the mug price). The quantity cue made things worse: "ferritin bumped up to 38" was
promoted above the prescription, so the cue was removed. The purchase cue never fired. Gemini batch 3
is now development data.

### Fixes 1–2 (2026-11-22, development on batches 1–3)

- **Answer conflicts:** a later memory that answers differently is served first, or the earlier one when the question asks "originally".
- **Other people's clauses:** hidden from the reader.

| Setup | gemini1 | gemini2 | gemini3 | traps declined (all) | dev |
|---|---|---|---|---|---|
| both fixes off (lowercase reading off) | 0.88 | 0.84 | 0.80 | 29/30 | 0.93 |
| both fixes on (lowercase reading off, the default) | 0.88 | 0.84 | 0.88 | 29/30 | 0.93 |
| both fixes on + lowercase reading | 1.00 | 0.88 | 0.96 | 27/30 | 0.93 |

The figures are hit@1. All three batches were tuned on, so none of these are claims; gemini4 confirms or rejects.
Recall latency on these batches: p50 48 → 61 ms, p95 unchanged (about 77 ms).
