# Recallix evaluation

Every number published in `eval/results/` (and on the website's Benchmarks page)
is produced by the commands below. Nothing is entered by hand.

## Data

| Split | Purpose |
|---|---|
| `data/dev/` | Fitting and tuning (importance weights, ranker weights + abstention threshold, rules). |
| `data/test/` | Held out. Reported, never used to choose patterns, cue words, seeds or weights. |
| `data/legacy/` | Pre-existing benchmarks ported from the old `scratch/` scripts (adversarial gauntlet v1–v3, 500-turn retention). Never tuned on. |

Tasks: `facts` (slot extraction P/R/F1), `schemas` (top-1 accuracy), `importance` (ROC AUC,
accuracy at 5.0 over an ordered conversation), `retrieval` (hit@1/hit@3 on answerable
questions, stale-top, false-abstention rate, and the share of unanswerable questions
correctly declined), `legacy`, `example` (one real store/recall run shown on the website).

**Caveat.** The sets are small (tens of items per task) and were written by the same author
as the extraction rules, so they can flatter the system. New test examples written by someone
else, in their own phrasing, are the most valuable contribution.

## Reproduce

```bash
pip install -r requirements-dev.txt -r requirements-local.txt

# vector engine benchmarks (single thread; recall measured against exact search)
cmake -S backend/infra_cpp -B backend/infra_cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build backend/infra_cpp/build -j
B=backend/infra_cpp/build/benchmark_hnsw
for cfg in "100000 128 16 100 50 clustered 500" "100000 128 16 100 50 uniform 500" "100000 128 16 100 200 uniform 300" \
           "100000 384 16 100 50 clustered 300" "1000000 128 16 100 50 clustered 200"; do
  bash -c "$B $cfg" | grep '^RESULT ' | sed 's/^RESULT //'
done > eval/results/hnsw.jsonl

# reference implementation on the same data distribution
pip install hnswlib
for cfg in "100000 128 clustered 50" "100000 128 uniform 50" "100000 128 uniform 200"; do
  bash -c "python eval/hnsw_reference.py $cfg"
done > eval/results/hnsw_reference.jsonl

# memory pipeline, against the real engine
export RECALLIX_INTERNAL_KEY=$(python -c "import secrets;print(secrets.token_hex(16))") RECALLIX_ENGINE_PORT=18431
backend/infra_cpp/build/memory_service & 
for split in dev test; do
  for e in hash st:sentence-transformers/all-MiniLM-L6-v2; do
    python eval/run_eval.py --split $split --embedder $e --engine-url http://127.0.0.1:18431 \
      --only facts,schemas,importance,retrieval,legacy,example \
      --json eval/results/${split}_$(echo $e | sed 's/st:sentence-transformers\///; s/[:\/]/_/g').json
  done
done
kill %1

python eval/publish_results.py     # -> eval/results/summary.json + RESULTS.md
```

Without `--engine-url` the harness uses an in-process brute-force engine (hermetic; used by CI).
`-v` prints every failure.

## Refitting the shipped models

```bash
python eval/run_eval.py --fit-importance --embedder st:sentence-transformers/all-MiniLM-L6-v2
python eval/run_eval.py --fit-ranker     --embedder st:sentence-transformers/all-MiniLM-L6-v2
```

Both start from fixed, documented priors (`backend/brain/memory_ranking/*.prior.json`) and write
`*.default.json` with provenance. The ranker chooses its regularisation strength and abstention
threshold by leave-one-scenario-out cross-validation on dev. In production, refit from real usage
instead (`fit_importance --from-db`, `fit_ranker --from-db`; see docs/quickstart.md).

## CI

`tests/test_eval_gate.py` reruns the hermetic dev eval (hash embedder) and fails if any metric
falls below the published `results/dev_hash.json`.

## Slot conventions

* `user` is the speaker: `user::name`, `user::location`, `user::employer`, `user::role`,
  `user::age`, `user::birthday`, `user::allergy`, `user::<pet>`, `user::favorite_<thing>`.
* Named entities are lowercased with non-alphanumerics removed (`Project Kestrel` → `projectkestrel`);
  lowercase multi-word subjects use underscores (`board meeting` → `board_meeting`).
* Scheduled events use `<event>::when`; aliases use `<alias>::alias_of` = the canonical name.
