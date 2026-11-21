# Measured results

Generated 2026-11-21 14:39 UTC from commit `cfb6e8ac + uncommitted changes` on Apple M4 (Darwin). Regenerate with the commands in `eval/README.md`.

## Memory pipeline (eval harness)

| split | embedder | facts F1 | schema acc | importance AUC / acc@5 | retrieval hit@1 / hit@3 | false abstain | unanswerable declined | gauntlet | retention@1 |
|---|---|---|---|---|---|---|---|---|---|
| dev | st:sentence-transformers/all-MiniLM-L6-v2 | 0.96 | 0.97 | 1.00 / 0.97 | 1.00 / 1.00 | 0.00 | 0.75 | 4/12 | 7/8 |
| dev | hash | 0.96 | 0.52 | 0.99 / 0.81 | 0.86 / 0.86 | 0.14 | 0.75 | 4/12 | 7/8 |
| test | st:sentence-transformers/all-MiniLM-L6-v2 | 0.84 | 0.85 | 1.00 / 0.94 | 0.79 / 0.93 | 0.07 | 0.75 | 4/12 | 7/8 |
| test | hash | 0.84 | 0.42 | 0.97 / 0.83 | 0.57 / 0.57 | 0.36 | 0.75 | 4/12 | 7/8 |

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
