# Embedding model comparison (2026-11-23, development data only)

- `raw_dense_dev.json`: 8 models, dense search alone. Covers LoCoMo dev conversations, our dev scenarios and Gemini batches 1–3 (all development data).
- `<model>_{dev,fresh,locomo_dev}.json`: the full Recallix pipeline for four models. For each model, the importance weights, the ranker and the answer-check threshold were refitted on dev data with the documented commands (eval/README.md). The fitted files were not kept in the repo.
- e5-base ran with the same "query: " prefix on questions and memories, because the pipeline has one embedding call for both. Its raw result used "query: " / "passage: ".
- Code state: answer conflicts and other-people clauses on; lowercase reading off.
- No test split was used.
