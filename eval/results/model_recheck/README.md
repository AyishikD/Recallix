# Model recheck with the bge-small embedder (2026-11-28, development data only)

## Reader (answer check)

The readers were last compared on 2026-11-21 with the old embedder (all-MiniLM-L6-v2). Rerun with
`eval/run_eval.py --split dev --embedder st:BAAI/bge-small-en-v1.5 --only retrieval --fit-answer-check ... --locomo ...`
(our dev scenarios + LoCoMo dev conversations). The fit rewrites answer_check.default.json; the file was restored
afterwards, so the live threshold (-2.721) is unchanged. Output: reader_fit_bge-small.json.

At the policy threshold (at most 5% of answerable questions with evidence refused):

| reader | AUC | unanswerable declined |
|---|---|---|
| deepset/roberta-base-squad2 (current) | 0.732 | **33.9%** |
| deepset/deberta-v3-base-squad2 | 0.758 | 27.8% |
| deepset/tinyroberta-squad2 | 0.699 | 27.8% |
| deepset/minilm-uncased-squad2 | 0.680 | 19.2% |

roberta-base-squad2 stays the choice (most unanswerable declined at the same refusal rate). The refit threshold with
the current code is -2.323 (stricter than -2.721); not applied here. An uncased reader is not a fix for lowercase text.

## Verifier (recall.answer_check.verify, used by cautious mode)

`eval/verifier_compare.py`: LoCoMo dev recalled once; every question the reader answered (859) is scored by each
candidate on the memories the reader accepted. Each model's threshold keeps 92.3% (the current 350/379 rate) of real
questions answered from an evidence memory. Output: verifier_compare_locomo_dev.json.

| verifier | size | real answers from evidence kept | answered trick questions declined | other real answers declined |
|---|---|---|---|---|
| cross-encoder/qnli-electra-base (current) | 110M | 382 / 414 | 56 / 160 | 114 / 285 |
| cross-encoder/qnli-distilroberta-base | 82M | 382 / 414 | 36 / 160 | 90 / 285 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | 382 / 414 | **83 / 160** | 73 / 285 |

At the same strictness the small ms-marco relevance model declines 27 more answered trick questions and 41 fewer of
the other real answers (answers from a memory not marked as evidence; some right, some wrong). It is five times
smaller. The thresholds were chosen on the same data they are scored on, so this needs confirmation on held-out data
(with the threshold frozen) before cautious mode switches model.

## Verifier confirmation: LoCoMo test passes, texting-style data fails (not switched)

Thresholds frozen from the dev run, LoCoMo test (conversations 5-9; viewed before for other decisions, never for this
one), 846 answered questions (verifier_compare_locomo_test_frozen.json):

| verifier | real answers from evidence kept | answered trick questions declined | other real answers declined |
|---|---|---|---|
| qnli-electra-base (current) | 379 / 403 | 62 / 143 | 139 / 300 |
| qnli-distilroberta-base | 376 / 403 | 49 / 143 | 101 / 300 |
| ms-marco-MiniLM-L-6-v2 | **382 / 403** | **78 / 143** | **90 / 300** |

On LoCoMo the small model is better on all three. But cautious mode end to end on the Gemini retrieval batches
(lowercase texting-style messages, development data; cautious_gemini_*.json):

| cautious mode with | hit@1 | answer match | real questions wrongly declined | trick questions declined |
|---|---|---|---|---|
| qnli-electra-base | 0.800 | 0.760 | 0.16 | 30 / 30 |
| ms-marco-MiniLM-L-6-v2 | 0.627 | 0.573 | **0.36** | 30 / 30 |

The threshold set on LoCoMo (edited, capitalised dialogue) does not transfer to texting-style messages, the same
failure seen earlier with a re-ranker as a gate. Not switched: cautious mode keeps qnli-electra-base. A verifier choice
has to be checked on texting-style data as well as LoCoMo.
