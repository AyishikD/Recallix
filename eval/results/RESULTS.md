# Measured results

Generated 2026-11-21 20:13 UTC from commit `64969ea0` on Apple M4 (Darwin). Regenerate with the commands in `eval/README.md`.

## Memory pipeline (eval harness)

| split | embedder | facts F1 | schema acc | importance AUC / acc@5 | retrieval hit@1 / hit@3 | false abstain | unanswerable declined | gauntlet | retention@1 |
|---|---|---|---|---|---|---|---|---|---|
| dev | st:sentence-transformers/all-MiniLM-L6-v2 | 0.96 | 0.97 | 1.00 / 0.97 | 0.86 / 0.96 | 0.04 | 0.25 | 7/12 | 7/8 |
| dev | hash | 0.96 | 0.52 | 0.99 / 0.81 | 0.82 / 0.93 | 0.07 | 0.62 | 6/12 | 7/8 |
| test | st:sentence-transformers/all-MiniLM-L6-v2 | 0.93 | 0.85 | 1.00 / 0.94 | 0.50 / 0.71 | 0.29 | 0.50 | 7/12 | 7/8 |
| test | hash | 0.93 | 0.42 | 0.97 / 0.83 | 0.43 / 0.50 | 0.43 | 0.50 | 6/12 | 7/8 |

## Vector engine (single thread, recall vs exact search)

| impl | n | dim | data | M | efC | efS | recall@10 | search avg ms | p99 ms | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| recallix (NEON) | 100,000 | 128 | clustered | 16 | 100 | 50 | 0.980 | 0.082 | 0.116 | 8.8 |
| recallix (NEON) | 100,000 | 128 | uniform | 16 | 100 | 50 | 0.189 | 0.149 | 0.218 | 17.9 |
| recallix (NEON) | 100,000 | 128 | uniform | 16 | 100 | 200 | 0.472 | 0.449 | 0.528 | 18.3 |
| recallix (NEON) | 100,000 | 384 | clustered | 16 | 100 | 50 | 0.956 | 0.164 | 0.274 | 25.3 |
| recallix (NEON) | 1,000,000 | 128 | clustered | 16 | 100 | 50 | 0.921 | 0.173 | 0.648 | 126.6 |
| hnswlib | 100,000 | 128 | clustered | 16 | 100 | 50 | 0.971 | 0.068 | – | 7.8 |
| hnswlib | 100,000 | 128 | uniform | 16 | 100 | 50 | 0.190 | 0.139 | – | 30.0 |
| hnswlib | 100,000 | 128 | uniform | 16 | 100 | 200 | 0.480 | 0.463 | – | 29.7 |

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:sentence-transformers/all-MiniLM-L6-v2, real engine, answer check deepset/roberta-base-squad2; no LLM)

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.608 | 0.696 | 0.745 | 0.887 | 0.548 | 0.821 | 0.550 | 0.930 | 0.589 |
| recallix_ranked | 0.713 | 0.833 | 0.911 | 0.921 | 0.715 | 0.982 | 0.583 | 0.961 | 0.825 |
| dense | 0.578 | 0.742 | 0.864 | 0.829 | 0.598 | 1.000 | 0.667 | 0.914 | 0.646 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallix declined 60.0% of 30 unanswerable questions and wrongly declined 20.0% of 470 answerable ones (AUC: answer margin 0.803, top relevance 0.639).

## LoCoMo retrieval, dev split (5 conversations, 2760 turns, 999 questions; embedder st:sentence-transformers/all-MiniLM-L6-v2, real engine; no LLM) — dev conversations were used for fitting

| system | recall@5 | recall@10 | hit@10 | multi-hop recall@10 | temporal recall@10 | open-domain recall@10 | single-hop recall@10 |
|---|---|---|---|---|---|---|---|
| recallix | 0.569 | 0.641 | 0.705 | 0.332 | 0.794 | 0.303 | 0.724 |
| recallix_ranked | 0.573 | 0.645 | 0.711 | 0.338 | 0.801 | 0.314 | 0.726 |
| dense | 0.440 | 0.540 | 0.611 | 0.330 | 0.654 | 0.282 | 0.597 |
| bm25 | 0.486 | 0.561 | 0.616 | 0.233 | 0.722 | 0.273 | 0.642 |

Recallix declined 0.8% of 237 adversarial (unanswerable) questions and wrongly declined 0.4% of 762 answerable ones.

## LoCoMo retrieval, test split (5 conversations, 3122 turns, 987 questions; embedder st:sentence-transformers/all-MiniLM-L6-v2, real engine; no LLM) — held out from fitting, but evaluated repeatedly during development

| system | recall@5 | recall@10 | hit@10 | multi-hop recall@10 | temporal recall@10 | open-domain recall@10 | single-hop recall@10 |
|---|---|---|---|---|---|---|---|
| recallix | 0.510 | 0.593 | 0.667 | 0.332 | 0.615 | 0.327 | 0.700 |
| recallix_ranked | 0.512 | 0.595 | 0.670 | 0.334 | 0.615 | 0.327 | 0.702 |
| dense | 0.355 | 0.458 | 0.532 | 0.284 | 0.458 | 0.257 | 0.536 |
| bm25 | 0.449 | 0.527 | 0.587 | 0.208 | 0.543 | 0.293 | 0.651 |

Recallix declined 1.0% of 209 adversarial (unanswerable) questions and wrongly declined 0.0% of 778 answerable ones.
