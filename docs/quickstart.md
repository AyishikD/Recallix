# Quickstart

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # API + memory pipeline
pip install -r requirements-local.txt      # optional: offline embeddings (embedding_provider: local)

cmake -S backend/infra_cpp -B backend/infra_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/infra_cpp/build -j
```

## 2. Configure

```bash
python -m backend.tools.setup_env
```

This writes a random `RECALLIX_INTERNAL_KEY` (the API ↔ engine secret) to `.env`,
creates the storage directory, and prints an **admin API key once**.

Choose the embedding model in `config.yaml` → `models.embedding_provider`:
`ollama` (default, `ollama pull mxbai-embed-large`), `local`
(sentence-transformers in-process, no server), or an OpenAI-compatible API.
No LLM is needed by Recallix itself.

## 3. Run

```bash
set -a; source .env; set +a
./backend/infra_cpp/build/memory_service &     # vector engine on 127.0.0.1:8080
python server.py                               # API on 127.0.0.1:8000
```

On startup the API rebuilds the engine's in-memory indexes from SQLite, so the
engine can be restarted at any time.

## 4. Create keys

```bash
python -m backend.tools.keys create --user alice                 # acts only as alice
python -m backend.tools.keys create --user my-app --role service  # may pass user_id for its end users
python -m backend.tools.keys list
python -m backend.tools.keys revoke rx_AbCd
```

## 5. Use

```bash
pip install -e sdk/python
```

```python
import os
from memoryos import Memory

mem = Memory(api_key=os.environ["RECALLIX_API_KEY"])
mem.store("My name is Priya and I live in Pune.")
out = mem.recall("Where do I live?")
print(out["status"], [m["content"] for m in out["memories"]])

# Tell Recallix which results were useful; this trains the ranker
# (python -m backend.brain.memory_ranking.fit_ranker --from-db)
mem.feedback(out["recall_id"], [m["id"] for m in out["memories"][:1]])
```

`status` is `CONSISTENT` (answer found), `ABSTAINED` (none of the matching memories
answers the question — Recallix returns nothing rather than a guess) or `NO_MATCH`.

The "does it answer?" decision is made by a small local reading-comprehension model
(SQuAD 2.0, no LLM) when `requirements-local.txt` is installed
(`recall.answer_check` in config.yaml; first use downloads the model, ~500 MB).
Without it, Recallix falls back to a learned relevance threshold.

TypeScript: `sdk/typescript` (`npm install ./sdk/typescript && npm run build` inside it).

## 6. Maintenance

```bash
python -m backend.tools.reindex              # rebuild indexes that disagree with SQLite
python -m backend.tools.reindex --backfill   # after changing the embedding model
python -m backend.brain.memory_ranking.fit_importance --from-db --dry-run
python -m backend.brain.memory_ranking.fit_ranker --from-db --dry-run
```
