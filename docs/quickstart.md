# Quickstart

## Fastest: Docker (one command)

```bash
git clone https://github.com/Recallic/Recallic && cd Recallic
docker compose up -d
docker compose logs recallic | grep -A1 "admin API key"     # printed once, on first start: save it
docker compose exec recallic python -m backend.tools.keys create --user alice   # a key for your app
```

The API is now on `http://localhost:8000` (bound to localhost only). One container runs the
vector engine and the API; all data, keys and downloaded models live in the `recallic-data`
volume, so restarts and upgrades keep everything. The first start downloads the embedding and
answer-check models (~600 MB) before it accepts requests; later starts take seconds. Nothing
leaves your machine after that: no LLM, no API calls.

Upgrade: `git pull && docker compose up -d --build`. To serve other machines, put a TLS
reverse proxy (Caddy, nginx) in front of port 8000 and set `CORS_ALLOWED_ORIGINS` in
`docker-compose.yml` for browser clients.

Then skip to [5. Use](#5-use).

## Without Docker

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt      # API + memory pipeline + local embeddings and answer check
                                           # (requirements.txt alone: no torch; set another embedding provider)
python -m backend.tools.get_engine         # prebuilt vector engine for this machine
```

`get_engine` downloads the engine built for your machine from the latest
[GitHub release](https://github.com/Recallic/Recallic/releases) (Linux x86-64 with or without
AVX2, Linux ARM64, macOS on Apple Silicon), checks it against the release's `SHA256SUMS`, and
installs it as `backend/infra_cpp/build/memory_service`. The Linux builds need glibc 2.35 or
newer (Ubuntu 22.04+, Debian 12+). On other platforms, or to compile it yourself:

```bash
cmake -S backend/infra_cpp -B backend/infra_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/infra_cpp/build -j
```

### 2. Configure

```bash
python -m backend.tools.setup_env
```

This writes a random `RECALLIC_INTERNAL_KEY` (the API ↔ engine secret) to `.env`,
creates the storage directory, and prints an **admin API key once**.

Choose the embedding model in `config.yaml` → `models.embedding_provider`: `local` (default:
sentence-transformers in-process, no server, model `BAAI/bge-small-en-v1.5`, the one the shipped
ranking, answer-check and trust fits were made with), `ollama`, or an OpenAI-compatible API.
No LLM is needed by Recallic itself.

The shipped fits belong to that model. With another embedding model, recall still works, but
refit on your own data for best results (`eval/README.md`), and after changing the model on an
existing deployment run `python -m backend.tools.reindex --backfill` so stored memories are
embedded again.

### 3. Run

```bash
set -a; source .env; set +a
./backend/infra_cpp/build/memory_service &     # vector engine on 127.0.0.1:8080
python server.py                               # API on 127.0.0.1:8000
```

On startup the API rebuilds the engine's in-memory indexes from SQLite, so the
engine can be restarted at any time. `curl http://127.0.0.1:8000/health` checks that both are up
(no key needed).

The first store or recall after starting takes up to a minute: the local models load then
(and download once, the very first time). Later requests take milliseconds. On a fresh
MacBook (Apple Silicon) the first store took about 30 seconds.

### 4. Create keys

```bash
python -m backend.tools.keys create --user alice                 # acts only as alice
python -m backend.tools.keys create --user my-app --role service  # may pass user_id for its end users
python -m backend.tools.keys list
python -m backend.tools.keys revoke rx_AbCd
```

## 5. Use

**From the browser:** open http://localhost:8000/docs (interactive API explorer), click
**Authorize**, paste your key, and try any endpoint with "Try it out". The same API is
described for other tools at http://localhost:8000/openapi.json.

**From code:**

```bash
pip install recallic                                       # Python SDK + recallic-mcp
```

```python
import os
from recallic import Memory

mem = Memory(api_key=os.environ["RECALLIC_API_KEY"])
mem.store("My name is Priya and I live in Pune.")
out = mem.recall("Where do I live?")
print(out["status"], [m["content"] for m in out["memories"]])

# One line of feedback: the first memory answered it (or name the ones you used: mem.used(out, m)).
# Recallic learns from this on its own; see "Learning from use" below.
mem.used(out)
```

`status` is `CONSISTENT` (answer found), `ABSTAINED` (none of the matching memories
answers the question — Recallic returns nothing rather than a guess) or `NO_MATCH`.
If your agent checks `status`, it can ask for the results even when recall declines:
`mem.recall(question, keep_on_decline=True)` (TypeScript: `{ keepOnDecline: true }`). `status`
stays `ABSTAINED` and `answer` is empty, so show them as "possibly related", not as the answer.
Off unless asked for (server-wide: `recall.keep_results_on_decline`).

For questions whose answer is spread over several things the user said ("what activities has
she done with her kids?"), ask for a deep recall: `mem.recall(question, deep=True)`. A small
local re-ranker (no LLM) then re-reads 50 candidates together with the question. It finds
more of the evidence for such questions (measurements: `eval/results/retrieval_fixes/`) and
costs about half a second per recall on a laptop CPU. `recall.rerank.enabled: true` makes every
recall deep.

**How cautious recall is** is one choice, per request or server-wide (`recall.mode`):

| `mode` | Use it when | What changes |
|---|---|---|
| `balanced` (default) | most apps | nothing extra |
| `cautious` | a wrong answer is far worse than "I don't know" (health, finance, legal) | every answer is checked a second time by a small local model; many more trick questions are declined, and more real questions too |
| `helpful` | refusals hurt more than the occasional miss | counting, ordering and "how long between" questions the usual check declined are answered when the memories support every event they take for granted; for time and ordering questions `computed_answer` gives the worked-out result ("41 days", or the order) with the dates in `receipt.computed` |

```python
out = mem.recall("How many days after I booked the flights did I cancel them?", mode="helpful")
out["computed_answer"]["text"]     # "21 days (3 weeks)", when every event is clearly in memory
```

`deep` and `keep_on_decline` work in every mode. Measured trade-offs (LoCoMo, LongMemEval):
[`eval/results/aggregation_check/README.md`](../eval/results/aggregation_check/README.md).

The "does it answer?" decision is made by a small local reading-comprehension model
(SQuAD 2.0, no LLM) when `requirements-local.txt` is installed
(`recall.answer_check` in config.yaml; first use downloads the model, ~500 MB).
Without it, Recallic falls back to a learned relevance threshold.

TypeScript (Node 18+): `npm install recallic`, then `import { Memory } from "recallic"`.

MCP (Claude Desktop, Cursor, ...): the Python package installs a `recallic-mcp` command; see
[Your memory, anywhere](#8-your-memory-anywhere).

## 6. Maintenance

```bash
python -m backend.tools.reindex              # rebuild indexes that disagree with SQLite
python -m backend.tools.reindex --backfill   # after changing the embedding model
python -m backend.brain.memory_ranking.fit_importance --from-db --dry-run
python -m backend.brain.memory_ranking.fit_ranker --from-db --dry-run
```

## 7. Trust tools

Recallic stores and recalls without an LLM, so the same messages always produce the same
memory. These tools build on that.

```bash
# Memory tests: assert what your agent remembers (and must not invent) — run them in CI.
python -m backend.tools.memtest tests/memory/*.yaml --embedder hash --answer-check off

# Replay and diff: rebuild memory from the original messages and see what would change
# (after editing rules, changing the embedding model, or just to prove determinism).
python -m backend.tools.replay snapshot --out now.json
python -m backend.tools.replay replay   --out new.json
python -m backend.tools.replay diff now.json new.json      # exit 1 if they differ

# Calibrate the answer check for this deployment, with a guarantee:
# "wrongly declines at most alpha of answerable questions, with probability >= 1 - delta".
python -m backend.tools.calibrate --from-file sample.json --alpha 0.05 --delta 0.05
python -m backend.tools.calibrate --from-db                 # from your users' feedback
```

A memory-test spec looks like this (see `tests/memory/example.yaml`):

```yaml
- name: changed facts stay current
  store:
    - "Project Alpha deadline is December 12."
    - "The deadline was moved to December 18."
  expect:
    - recall: "When is the Project Alpha deadline?"
      top_contains: "December 18"
    - fact: projectalpha::deadline
      contains: "December 18"
```

Every recall also returns a `receipt`: why it answered or declined, the versions of the
ranker, embedding model, extraction rules and answer check involved, the calibration
guarantee in force, and for each memory where its facts came from (rule and exact quote).

## 8. Your memory, anywhere

**Export / import** in an open format (also from Mem0 exports or plain text):

```bash
python -m backend.tools.portable export --user alice --out alice.json
python -m backend.tools.portable import --user alice --file mem0.json --format mem0
```

or over the API: `GET /memory/export`, `POST /memory/import`, and in the SDKs
`mem.export()` / `mem.import_memories(data, format="mem0")`.

**MCP server**: give Claude Desktop, Cursor or any MCP client the same memory. The Python
package installs a `recallic-mcp` command (or use `python -m recallic.mcp_server`):

```json
"mcpServers": {
  "recallic": {
    "command": "recallic-mcp",
    "env": {"RECALLIC_API_KEY": "rx_...", "RECALLIC_AGENT_ID": "claude-desktop"}
  }
}
```

Tools: `recall_memory`, `store_memory`, `confirm_answer`, `relevant_memories`, `update_memory`, `forget_memory`, `export_memories`.
Memories stored as `shared` are visible to every client and agent of the same user.

## 9. Find problems before your users do

```bash
# Memory fuzzer: generates tests from the stored facts — asked differently, asked about a
# stranger, and changed — and writes every failure as a runnable memory test.
python -m backend.tools.fuzz --user alice --write-failures failures.yaml

# Deletion log: every delete returns a receipt; this checks no entry was altered afterwards.
python -m backend.tools.audit verify
```

Each recall reports `trust` (a "this answers it" probability), each store reports
`conflicts` (facts the message changed), and `POST /memory/proactive` returns memories worth
keeping in mind for a message nobody asked a question about. In the SDK: `mem.proactive(message)`.

The trust score ships fitted on LoCoMo. It stays calibrated on new conversations of that kind,
but not on every kind (on LongMemEval it understated how often answers were right), so
recalibrate it once your users have confirmed about 50 answers (feedback / `confirm_answer`):

```bash
python -m backend.tools.calibrate_trust --from-db          # or --from-file sample.json
```

It refits only the score's scale (two numbers), keeps the result only if it improves held-out
accuracy, and from then on `receipt.trust_calibration` says `this deployment`.
In a simulated deployment (fitted on about a hundred answers from other LongMemEval questions), this
brought the score back within the calibration bar: eval/results/trust_check_longmemeval_recalibrated.json.

## Learning from use

Each `mem.used(out)` (or `POST /memory/feedback`) records which recalled memories were useful. Once a
day (`workers.learning_hours`), Recallic refits its ranking from that feedback and **keeps the new fit
only if it ranks better on feedback it was not fitted on** (`learning` in `config.yaml`):

* nothing happens until 30 recalls have feedback, and then only when 20 new ones have arrived;
* a fifth of the feedback is set aside; the new fit must put the right memory first at least as often,
  in the first five at least as often, be strictly better on one of the two, and must not decline
  answerable questions more often than allowed;
* the model it replaces is kept, and every attempt is logged.

```bash
curl -H "X-API-Key: $ADMIN_KEY" localhost:8000/internal/learning             # what it has tried and adopted
curl -X POST -H "X-API-Key: $ADMIN_KEY" localhost:8000/internal/learning/rollback
python -m backend.brain.memory_ranking.self_improve --dry-run                # what a refit would do now
```

It uses only this deployment's own feedback; nothing leaves your server. To turn it off: `learning.enabled: false`.

**Optional: learn when to say "I don't know"** (`learning.answer_check.enabled: true`, off by default).
The answer check declines below a threshold, and the right threshold depends on your users' questions:
where nearly everything asked has an answer in memory it should decline less, where many questions ask
about things never said it should decline more. With this on, the same nightly run also picks the
threshold from feedback: recalls where a memory was used count as answerable, `mem.none_used(out)` as
not answerable, and a wrong answer counts as 3 refusals (`wrong_answer_cost`). A new threshold is kept
only if it scores better on held-out feedback, and needs 50 answerable and 20 not-answerable recalls first.
To let it learn from wrongly declined questions too, recall with `keep_on_decline=True` and give feedback
on declined recalls as well.

```bash
curl -X POST -H "X-API-Key: $ADMIN_KEY" "localhost:8000/internal/learning/rollback?target=answer_check"
python -m backend.brain.features.answer_check_learning --dry-run   # what it would choose now
```
