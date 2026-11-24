# Recallix

Persistent memory for AI agents: store what a user tells any of their agents, recall it
later by meaning, keep facts up to date, and say "I don't know" instead of guessing.

[![Watch the 50-second demo: a week of messy texting, three answers, and one honest decline](public/videos/recallix-demo-poster.jpg)](https://recallix.vercel.app/videos/recallix-demo.mp4)

*50-second demo, narrated (click to play). Every message, answer and timing in it comes from a real run.*

* **No LLM in the memory pipeline.** Fact extraction, schema tagging, importance, ranking,
  reflections and predictions are rules, embeddings and small fitted models. Recallix needs
  only an embedding model (Ollama, in-process sentence-transformers, or an API).
* **Per-user HNSW vector engine in C++** (NEON / AVX2), rebuilt from SQLite on start, so
  the engine can restart at any time without losing anything.
* **Learned, explainable recall.** Each result carries its relevance and the feature
  breakdown behind it. Weights and the abstention threshold are fitted on labelled data and
  can be refit from your users' feedback.

## Measured results

Copied from [`eval/results/RESULTS.md`](eval/results/RESULTS.md) (generated 2026-11-21 on an
Apple M4; regenerate with the commands in [`eval/README.md`](eval/README.md)). The website's
Benchmarks page reads the same file.

**Memory pipeline**: held-out test split, `all-MiniLM-L6-v2` embeddings, real engine:

| Metric | Result |
|---|---|
| Recall hit@1 / hit@3 (answerable questions) | 0.79 / 0.93 |
| Unanswerable questions correctly declined | 3/4 (false abstention on answerable: 0.07) |
| Fact extraction F1 | 0.84 |
| Schema accuracy | 0.85 |
| Importance ROC AUC | 1.00 |
| Recall / store latency, p50 (incl. embedding) | 8.3 ms / 9.2 ms |
| Legacy adversarial gauntlet v1–v3 | 4/12 |
| Legacy 500-turn retention, recall@1 | 7/8 |

**LoCoMo** (independent public benchmark: 10 conversations, 5,882 turns, 1,986 questions; never
used for tuning). It measures how much of each question's labelled evidence is retrieved, with
no LLM involved:

| recall@10 | overall | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|---|
| **Recallix** | **0.591** | **0.317** | **0.643** | **0.342** | **0.690** |
| Dense vectors (same embeddings) | 0.499 | 0.307 | 0.554 | 0.269 | 0.566 |
| BM25 | 0.544 | 0.221 | 0.630 | 0.283 | 0.647 |

Abstention does **not** transfer to LoCoMo's adversarial questions: Recallix declined only 0.4%
of them. They ask about topics the conversation did discuss (attributing an event to the wrong
person), which a retrieval-confidence threshold cannot detect; that needs an answer-level check.
Recallix also wrongly declined only 0.4% of answerable questions.

The legacy gauntlet and retention benchmarks are the pre-existing ones from this repository.
Earlier versions of this README reported 10/12 and 75% for them. Those scores depended on
answers hardcoded into the recall code (e.g. a fixed alias table), which has been removed.

**Vector engine** (single thread, recall@10 against exact search, M=16, efConstruction=100, efSearch=50):

| Vectors | Dim | Data | Recall@10 | Search avg / p99 | Build |
|---|---|---|---|---|---|
| 100,000 | 128 | clustered | 0.980 | 0.082 / 0.116 ms | 8.8 s |
| 100,000 | 384 | clustered | 0.956 | 0.164 / 0.274 ms | 25.3 s |
| 1,000,000 | 128 | clustered | 0.921 | 0.173 / 0.648 ms | 127 s |
| 100,000 | 128 | uniform random | 0.189 (hnswlib: 0.190) | 0.149 ms | 17.9 s |

Uniform random vectors are the known worst case for HNSW; the reference implementation
(hnswlib) gets the same recall on the same data, which validates the engine. These numbers exclude
the Python API and the embedding model.

**Limits worth knowing:**
* The eval sets are small and were written by the same author as the extraction rules.
* Fact extraction is rule-based. It misses phrasings such as "we have a…", "rescheduled to"
  and "also called"; unextracted facts are still stored and searchable, just not versioned.

## Architecture

```
client ──X-API-Key──▶ FastAPI (server.py)
                        ├─ store:  dedupe → embed once → facts → schema → importance
                        │          → SQLite (row + embedding + FTS5) → engine /add
                        ├─ recall: aliases → FTS5 + entity index + user's HNSW → learned ranker
                        │          → truth arbitration → multi-hop → abstain below threshold
                        ├─ index sync: SQLite → engine (startup, engine restart, reindex)
                        └─ jobs: world state · next-topic prediction · reflections · forgetting
                      SQLite (system of record)        C++ engine (one HNSW index per user)
```

Diagrams: [`docs/architecture_high_level.puml`](docs/architecture_high_level.puml),
[`docs/architecture_technical_depth.puml`](docs/architecture_technical_depth.puml).

## Getting started

**Docker (recommended):**

```bash
git clone https://github.com/AyishikD/Recallix && cd Recallix
docker compose up -d                                       # engine + API on localhost:8000
docker compose logs recallix | grep -A1 "admin API key"    # printed once: save it
docker compose exec recallix python -m backend.tools.keys create --user alice
pip install recallix                                       # Python SDK + recallix-mcp
npm install recallix                                       # TypeScript / Node 18+
```

```python
from recallix import Memory
mem = Memory(api_key="rx_...")
mem.store("I moved to Pune last month.")
print(mem.recall("Where do I live?")["memories"][0]["content"])
```

**From source:**

```bash
pip install -r requirements-local.txt           # API + local embeddings and answer check
cmake -S backend/infra_cpp -B backend/infra_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/infra_cpp/build -j

python -m backend.tools.setup_env               # writes the engine secret to .env, prints an admin key
set -a; source .env; set +a
./backend/infra_cpp/build/memory_service &      # vector engine
python server.py                                # API on 127.0.0.1:8000
```

Then create a key (`python -m backend.tools.keys create --user alice`) and use the
[Python SDK](sdk/python) or [TypeScript SDK](sdk/typescript). Full walkthrough:
[`docs/quickstart.md`](docs/quickstart.md). Endpoints: [`docs/api_reference.md`](docs/api_reference.md).
All settings live in [`config.yaml`](config.yaml). Secrets come only from the environment.

## Security model

* API keys are random tokens stored only as SHA-256 hashes, with roles: `user` (acts as itself),
  `service` (may act for its end users), `admin` (maintenance endpoints).
* Each user has a separate vector index; private memories are visible only to the agent that
  wrote them, enforced in SQL.
* Deleting a memory removes its vector, long-term copy and timeline entry. Clearing a user
  removes their data from every store.
* The engine binds to localhost and refuses to start without `RECALLIX_INTERNAL_KEY`.

## Development

```bash
pip install -r requirements-dev.txt
RECALLIX_ENGINE_BIN=backend/infra_cpp/build/memory_service python -m pytest tests
ctest --test-dir backend/infra_cpp/build --output-on-failure   # includes a ThreadSanitizer run
python eval/run_eval.py --split dev -v
```

CI (`.github/workflows/backend.yml`) builds the engine on Linux (AVX2) and macOS (NEON), runs
the C++ tests under ThreadSanitizer, the API/integration tests against the real engine binary,
and an eval regression gate.

`examples/multi_agent_handover.py` is the multi-agent demo (the agents use an LLM;
Recallix does not).

## License

- **Recallix (server, engine, everything in this repository except the SDKs):**
  [Functional Source License 1.1, ALv2 Future License](LICENSE) (FSL-1.1-ALv2).
  - **What you may do:** use, modify and self-host it for anything, including inside your own commercial
    products.
  - **What you may not do:** offer it, or a substantially similar product built from it, as a competing
    commercial product or service.
  - **Each version becomes Apache 2.0** two years after release.
- **Client SDKs** ([`sdk/python`](sdk/python), [`sdk/typescript`](sdk/typescript)), published as `recallix` on PyPI and
  npm: [Apache 2.0](sdk/python/LICENSE).
- **Earlier releases:** versions up to commit `e5fddcd0` (2026-11-24) were released under Apache 2.0 and stay available
  under it.

"Recallix" and its logo are trademarks of Ayishik Das; see [NOTICE](NOTICE).
