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

# LoCoMo (public benchmark, retrieval only, no LLM)
mkdir -p /tmp/locomo && curl -sSfL -o /tmp/locomo/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
for split in test dev; do   # dev = conversations 1-5 (used for fitting), test = 6-10
  python eval/benchmarks/locomo.py --split $split --data /tmp/locomo/locomo10.json \
    --embedder st:sentence-transformers/all-MiniLM-L6-v2 --engine-url http://127.0.0.1:18431 \
    --json eval/results/locomo_${split}_all-MiniLM-L6-v2.json
done
kill %1

python eval/publish_results.py     # -> eval/results/summary.json + RESULTS.md
```

Without `--engine-url` the harness uses an in-process brute-force engine (hermetic; used by CI).
`-v` prints every failure.

## Refitting the shipped models (dev data only, in this order)

```bash
E=st:sentence-transformers/all-MiniLM-L6-v2; L=/tmp/locomo/locomo10.json
python eval/run_eval.py --fit-importance --embedder $E
RECALLIX_ANSWER_CHECK=0 python eval/run_eval.py --fit-ranker --locomo $L --embedder $E
python eval/run_eval.py --fit-answer-check deepset/minilm-uncased-squad2,deepset/tinyroberta-squad2,deepset/roberta-base-squad2,deepset/deberta-v3-base-squad2 \
    --locomo $L --embedder $E          # then set recall.answer_check.model in config.yaml to the chosen model
```

* Importance and ranker fits start from fixed, documented priors (`backend/brain/memory_ranking/*.prior.json`)
  and write `*.default.json` with provenance. The ranker uses our dev scenarios plus LoCoMo **dev**
  conversations (1-5), chooses its regularisation by leave-one-group-out CV on hit@5, and sets its
  relevance threshold so at most `recall.max_false_abstain_rate` of answerable dev questions are refused.
* The answer check (`backend/brain/features/answer_check.py`) compares SQuAD 2.0 readers on the same dev data;
  its threshold uses the same 5% wrong-refusal policy; the model that then declines the most unanswerable dev
  questions is chosen (`backend/brain/features/answer_check.default.json` records every candidate's numbers).
* In production, refit from real usage instead (`fit_importance --from-db`, `fit_ranker --from-db`).

## Fresh held-out benchmark: LongMemEval

Never used for fitting or design decisions; run once with the frozen configuration.

```bash
python -c "from huggingface_hub import hf_hub_download as d; print(d('xiaowu0162/longmemeval-cleaned', \
    'longmemeval_s_cleaned.json', repo_type='dataset', local_dir='/tmp/lme'))"
python eval/benchmarks/longmemeval.py --data /tmp/lme/longmemeval_s_cleaned.json --embedder $E \
    --engine-url http://127.0.0.1:18431 --json eval/results/longmemeval_s_all-MiniLM-L6-v2.json
```

## CI

`tests/test_eval_gate.py` reruns the hermetic dev eval (hash embedder, answer check off so it runs
without torch) and fails if any metric falls below `results/gate_dev_hash.json`, produced by:

```bash
RECALLIX_ANSWER_CHECK=0 python eval/run_eval.py --split dev --embedder hash \
    --only facts,schemas,importance,retrieval,legacy --json eval/results/gate_dev_hash.json
```

## Slot conventions

* `user` is the speaker: `user::name`, `user::location`, `user::employer`, `user::role`,
  `user::age`, `user::birthday`, `user::allergy`, `user::<pet>`, `user::favorite_<thing>`.
* Named entities are lowercased with non-alphanumerics removed (`Project Kestrel` → `projectkestrel`);
  lowercase multi-word subjects use underscores (`board meeting` → `board_meeting`).
* Scheduled events use `<event>::when`; aliases use `<alias>::alias_of` = the canonical name.

## Open harness: compare any memory system

`eval/harness.py` runs any memory system through the same benchmark with the same
scoring as the scripts above (verified: identical numbers to `benchmarks/locomo.py`).
A system is a four-method adapter (`eval/adapters/base.py`); built in are `recallix`
(what it serves, after its decline decision), `recallix-ranked`, `vector-search` and
`bm25`.

```bash
python eval/harness.py --dataset locomo --data locomo10.json --split dev \
    --systems recallix,recallix-ranked,vector-search,bm25 \
    --embedder st:sentence-transformers/all-MiniLM-L6-v2 --json results.json
```

Each result records every system's configuration and a fingerprint of the dataset file.
`eval/adapters/mem0_adapter.py` is **unverified** (written from public docs, not yet run:
it needs `mem0ai` and an LLM key) — do not publish its numbers until it has been run and
reviewed. Vendors are invited to submit or correct their own adapters.

