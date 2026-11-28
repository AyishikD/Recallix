# Measured results

Generated 2026-11-24 17:44 UTC from commit `d62eaee8` on Apple M4 (Darwin). Regenerate with the commands in `eval/README.md`.

## Memory pipeline (eval harness)

| file | split | embedder | facts F1 | schema acc | importance AUC / acc@5 | retrieval hit@1 / hit@3 | false abstain | unanswerable declined | gauntlet | retention@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dev_all-MiniLM-L6-v2` | dev | st:sentence-transformers/all-MiniLM-L6-v2 | 0.96 | 0.97 | 1.00 / 0.97 | 0.86 / 0.96 | 0.04 | 0.25 | 7/12 | 7/8 |
| `dev_bge-small-en-v1.5` | dev | st:BAAI/bge-small-en-v1.5 | 0.96 | 0.97 | 1.00 / 0.94 | 0.93 / 0.96 | 0.04 | 1.00 | 8/12 | 7/8 |
| `dev_hash` | dev | hash | 0.96 | 0.52 | 0.99 / 0.81 | 0.82 / 0.93 | 0.07 | 0.62 | 6/12 | 7/8 |
| `fresh_dev_batches1-2_all-MiniLM-L6-v2` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.92 / 0.92 | 0.08 | 0.90 | – | – |
| `fresh_dev_batches1-3_bge-small-en-v1.5` | fresh | st:BAAI/bge-small-en-v1.5 | – | – | – / – | 0.84 / 0.84 | 0.13 | 0.93 | – | – |
| `fresh_gemini1_all-MiniLM-L6-v2` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.76 / 0.88 | 0.08 | 0.90 | – | – |
| `fresh_gemini2_all-MiniLM-L6-v2` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.80 / 0.84 | 0.16 | 1.00 | – | – |
| `fresh_gemini2_all-MiniLM-L6-v2_changes_off` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.80 / 0.84 | 0.16 | 0.90 | – | – |
| `fresh_gemini3_all-MiniLM-L6-v2` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.80 / 0.96 | 0.04 | 0.80 | – | – |
| `fresh_gemini3_all-MiniLM-L6-v2_all_off` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.80 / 0.92 | 0.08 | 1.00 | – | – |
| `fresh_gemini3_all-MiniLM-L6-v2_before_reader_fix` | fresh | st:sentence-transformers/all-MiniLM-L6-v2 | – | – | – / – | 0.80 / 0.92 | 0.08 | 1.00 | – | – |
| `test_all-MiniLM-L6-v2` | test | st:sentence-transformers/all-MiniLM-L6-v2 | 0.93 | 0.85 | 1.00 / 0.94 | 0.50 / 0.71 | 0.29 | 0.50 | 7/12 | 7/8 |
| `test_bge-small-en-v1.5` | test | st:BAAI/bge-small-en-v1.5 | 0.93 | 0.85 | 1.00 / 0.94 | 0.86 / 0.93 | 0.07 | 0.75 | 8/12 | 7/8 |
| `test_hash` | test | hash | 0.93 | 0.42 | 0.97 / 0.83 | 0.43 / 0.50 | 0.43 | 0.50 | 6/12 | 7/8 |

`fresh` rows are retrieval-only runs on the practice batches. Other result files (not pipeline evals): `trust_check_locomo_test.json`, `trust_check_longmemeval.json`, `trust_check_longmemeval_recalibrated.json`.

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

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:sentence-transformers/all-MiniLM-L6-v2, real engine, answer check deepset/roberta-base-squad2; no LLM) — `longmemeval_s_all-MiniLM-L6-v2.json`

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.608 | 0.696 | 0.745 | 0.887 | 0.548 | 0.821 | 0.550 | 0.930 | 0.589 |
| recallix_ranked | 0.713 | 0.833 | 0.911 | 0.921 | 0.715 | 0.982 | 0.583 | 0.961 | 0.825 |
| dense | 0.578 | 0.742 | 0.864 | 0.829 | 0.598 | 1.000 | 0.667 | 0.914 | 0.646 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallic declined 60.0% of 30 unanswerable questions and wrongly declined 20.0% of 470 answerable ones (AUC: answer margin 0.803, top relevance 0.639).

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:BAAI/bge-small-en-v1.5, real engine, answer check deepset/roberta-base-squad2; no LLM) — `longmemeval_s_bge-small-en-v1.5.json`

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.651 | 0.727 | 0.789 | 0.928 | 0.559 | 0.804 | 0.744 | 0.953 | 0.622 |
| recallix_ranked | 0.745 | 0.860 | 0.940 | 0.963 | 0.748 | 0.964 | 0.717 | 0.977 | 0.838 |
| dense | 0.648 | 0.798 | 0.908 | 0.903 | 0.737 | 0.946 | 0.828 | 0.898 | 0.674 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallic declined 56.7% of 30 unanswerable questions and wrongly declined 16.6% of 470 answerable ones (AUC: answer margin 0.793, top relevance 0.673).

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:BAAI/bge-small-en-v1.5, real engine, answer check deepset/roberta-base-squad2; no LLM) — `longmemeval_s_bge-small-en-v1.5_before-retrieval-fixes.json`

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.637 | 0.732 | 0.783 | 0.949 | 0.596 | 0.786 | 0.667 | 0.945 | 0.623 |
| recallix_ranked | 0.743 | 0.859 | 0.940 | 0.963 | 0.748 | 0.964 | 0.717 | 0.977 | 0.834 |
| dense | 0.648 | 0.798 | 0.908 | 0.903 | 0.737 | 0.946 | 0.828 | 0.898 | 0.674 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallic declined 53.3% of 30 unanswerable questions and wrongly declined 18.3% of 470 answerable ones (AUC: answer margin 0.782, top relevance 0.672).

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:BAAI/bge-small-en-v1.5, real engine, answer check deepset/roberta-base-squad2; no LLM) — deep recall — `longmemeval_s_bge-small-en-v1.5_deep.json`

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.641 | 0.763 | 0.804 | 0.965 | 0.649 | 0.804 | 0.722 | 0.922 | 0.668 |
| recallix_ranked | 0.745 | 0.860 | 0.940 | 0.963 | 0.748 | 0.964 | 0.717 | 0.977 | 0.838 |
| dense | 0.648 | 0.798 | 0.908 | 0.903 | 0.737 | 0.946 | 0.828 | 0.898 | 0.674 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallic declined 56.7% of 30 unanswerable questions and wrongly declined 16.6% of 470 answerable ones (AUC: answer margin 0.793, top relevance 0.673).

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:BAAI/bge-small-en-v1.5, real engine, answer check deepset/roberta-base-squad2; no LLM) — results kept on decline (`recall.keep_results_on_decline`) — `longmemeval_s_bge-small-en-v1.5_keep-on-decline.json`

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.748 | 0.838 | 0.921 | 0.942 | 0.718 | 0.929 | 0.811 | 0.984 | 0.786 |
| recallix_ranked | 0.745 | 0.860 | 0.940 | 0.963 | 0.748 | 0.964 | 0.717 | 0.977 | 0.838 |
| dense | 0.648 | 0.798 | 0.908 | 0.903 | 0.737 | 0.946 | 0.828 | 0.898 | 0.674 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallic declined 56.7% of 30 unanswerable questions and wrongly declined 16.6% of 470 answerable ones (AUC: answer margin 0.793, top relevance 0.673).

## LongMemEval_S (fresh held-out: 500 questions, 246,750 turns; embedder st:BAAI/bge-small-en-v1.5, real engine, answer check deepset/roberta-base-squad2; no LLM) — deep recall — results kept on decline (`recall.keep_results_on_decline`) — `longmemeval_s_bge-small-en-v1.5_keep-on-decline_deep.json`

| system | recall@5 | recall@10 | hit@10 | knowledge-update recall@10 | multi-session recall@10 | single-session-assistant recall@10 | single-session-preference recall@10 | single-session-user recall@10 | temporal-reasoning recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| recallix | 0.738 | 0.874 | 0.936 | 0.979 | 0.808 | 0.929 | 0.789 | 0.953 | 0.833 |
| recallix_ranked | 0.745 | 0.860 | 0.940 | 0.963 | 0.748 | 0.964 | 0.717 | 0.977 | 0.838 |
| dense | 0.648 | 0.798 | 0.908 | 0.903 | 0.737 | 0.946 | 0.828 | 0.898 | 0.674 |
| bm25 | 0.660 | 0.767 | 0.851 | 0.889 | 0.636 | 0.911 | 0.444 | 0.930 | 0.753 |

Recallic declined 56.7% of 30 unanswerable questions and wrongly declined 16.6% of 470 answerable ones (AUC: answer margin 0.793, top relevance 0.673).

## LoCoMo retrieval, dev split (5 conversations, 2760 turns, 999 questions; embedder st:sentence-transformers/all-MiniLM-L6-v2, real engine; no LLM) — dev conversations were used for fitting

| system | recall@5 | recall@10 | hit@10 | multi-hop recall@10 | temporal recall@10 | open-domain recall@10 | single-hop recall@10 |
|---|---|---|---|---|---|---|---|
| recallix | 0.569 | 0.641 | 0.705 | 0.332 | 0.794 | 0.303 | 0.724 |
| recallix_ranked | 0.573 | 0.645 | 0.711 | 0.338 | 0.801 | 0.314 | 0.726 |
| dense | 0.440 | 0.540 | 0.611 | 0.330 | 0.654 | 0.282 | 0.597 |
| bm25 | 0.486 | 0.561 | 0.616 | 0.233 | 0.722 | 0.273 | 0.642 |

Recallic declined 0.8% of 237 adversarial (unanswerable) questions and wrongly declined 0.4% of 762 answerable ones.

## LoCoMo retrieval, dev split (5 conversations, 2760 turns, 999 questions; embedder st:BAAI/bge-small-en-v1.5, real engine; no LLM) — dev conversations were used for fitting

| system | recall@5 | recall@10 | hit@10 | multi-hop recall@10 | temporal recall@10 | open-domain recall@10 | single-hop recall@10 |
|---|---|---|---|---|---|---|---|
| recallix | 0.586 | 0.701 | 0.758 | 0.375 | 0.817 | 0.280 | 0.812 |
| recallix_ranked | 0.591 | 0.673 | 0.741 | 0.379 | 0.816 | 0.394 | 0.750 |
| dense | 0.512 | 0.602 | 0.660 | 0.420 | 0.736 | 0.339 | 0.641 |
| bm25 | 0.486 | 0.561 | 0.616 | 0.233 | 0.722 | 0.273 | 0.642 |

Recallic declined 32.5% of 237 adversarial (unanswerable) questions and wrongly declined 7.9% of 762 answerable ones.

## LoCoMo retrieval, test split (5 conversations, 3122 turns, 987 questions; embedder st:sentence-transformers/all-MiniLM-L6-v2, real engine; no LLM) — held out from fitting, but evaluated repeatedly during development

| system | recall@5 | recall@10 | hit@10 | multi-hop recall@10 | temporal recall@10 | open-domain recall@10 | single-hop recall@10 |
|---|---|---|---|---|---|---|---|
| recallix | 0.510 | 0.593 | 0.667 | 0.332 | 0.615 | 0.327 | 0.700 |
| recallix_ranked | 0.512 | 0.595 | 0.670 | 0.334 | 0.615 | 0.327 | 0.702 |
| dense | 0.355 | 0.458 | 0.532 | 0.284 | 0.458 | 0.257 | 0.536 |
| bm25 | 0.449 | 0.527 | 0.587 | 0.208 | 0.543 | 0.293 | 0.651 |

Recallic declined 1.0% of 209 adversarial (unanswerable) questions and wrongly declined 0.0% of 778 answerable ones.

## LoCoMo retrieval, test split (5 conversations, 3122 turns, 987 questions; embedder st:BAAI/bge-small-en-v1.5, real engine; no LLM) — held out from fitting, but evaluated repeatedly during development

| system | recall@5 | recall@10 | hit@10 | multi-hop recall@10 | temporal recall@10 | open-domain recall@10 | single-hop recall@10 |
|---|---|---|---|---|---|---|---|
| recallix | 0.535 | 0.659 | 0.727 | 0.363 | 0.650 | 0.340 | 0.793 |
| recallix_ranked | 0.542 | 0.619 | 0.694 | 0.368 | 0.638 | 0.338 | 0.724 |
| dense | 0.474 | 0.570 | 0.653 | 0.393 | 0.584 | 0.351 | 0.646 |
| bm25 | 0.449 | 0.527 | 0.587 | 0.208 | 0.543 | 0.293 | 0.651 |

Recallic declined 31.6% of 209 adversarial (unanswerable) questions and wrongly declined 9.0% of 778 answerable ones.
