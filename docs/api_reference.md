# API reference

All endpoints require `X-API-Key`. Responses are `{"status": "success", "data": ..., "timestamp": ...}`;
errors are HTTP 4xx/5xx with `{"detail": ...}`.

**Keys and users.** A `user` key acts only as its own user. `service` and `admin` keys may pass
`user_id` (body for POST, query string for GET/DELETE) to act for an end user; a `user` key
that does so gets 403. `/internal/*` needs an `admin` key.

**Agents.** `agent_id` defaults to `default_agent`. A memory is visible to an agent if it is
`shared` or that agent's own `private` memory.

## Health

`GET /health` needs no API key: `200 {"status": "ok", "api": "ok", "engine": "ok"}` when the API and the
vector engine both answer, `503` with `"engine": "unreachable"` when the engine does not (recall then falls back
to keyword search). For monitors, load balancers and container health checks; it reveals no user data.

## Memory

| Method | Path | Body / query | Notes |
|---|---|---|---|
| POST | `/memory/store` | `content` (≤ 20,000 chars), `agent_id?`, `memory_type? ("private"\|"shared")`, `metadata?` (≤ `api.max_metadata_bytes` as JSON), `user_id?`, `timestamp?` (Unix seconds, when it was originally said; default now; at most a day ahead) | Returns `memory_id`, `importance`, `schemas`, `status` (`stored` / `skipped_duplicate`) and `conflicts` — facts this message changes, with the previous value and when it was said (the newest value is kept). 429 above `api.writes_per_second_per_user`. |
| POST | `/memory/recall` | `query`, `limit? (1-100)`, `agent_id?`, `user_id?`, `deep?`, `keep_on_decline?` | Returns `recall_id`, `status` (`CONSISTENT` / `ABSTAINED` / `NO_MATCH`), `answer` (when an answer was found: `text`, the reader's most confident span; `memory_id`; `margin`; and `later_answer` / `later_memory_id` when a later memory gives a different answer, a possible update), `memories[]` each with `relevance`, an `explain` feature breakdown and, when the answer check is enabled, `answer` (the extracted span) and `answer_margin`. `debug.answer_check` shows the reader's decision. With `recall.aggregation_check` on (off by default), counting, ordering and time-between questions are decided by whether the memories support every event the question takes for granted; `debug.aggregation_check` lists those events with their support, and `needs_calculation: true` says the answer has to be worked out across the returned memories. On `ABSTAINED`, `memories` is empty unless `keep_on_decline: true` (per request; omitted = `recall.keep_results_on_decline`, off by default): then it holds the ranked results, low-confidence, and `answer` stays null. Only for callers that check `status`. `deep: true` re-reads the candidates after the first three with a local cross-encoder (`recall.rerank`): better when the answer is spread over several memories, slower (about 0.5 s on a laptop CPU); `false` never, omitted = the server setting (off by default). Memories it moved carry `rerank_score`. `mode` (`cautious` / `balanced` / `helpful`; omitted = `recall.mode`, `balanced` by default) chooses how readily recall answers: `cautious` also verifies every answer (declines far more trick questions, refuses more real ones), `helpful` also answers counting, ordering and time questions the usual check declined when the memories support every event they take for granted; `receipt.mode` and `debug.mode` say which applied. `receipt` explains the decision (`decision`, `reason`, `guarantee`, `trust`, `versions`); each memory carries `trust`, a probability that it answers the question (fitted on LoCoMo dev; recalibrate it for your deployment with `python -m backend.tools.calibrate_trust --from-db` once ~50 answers have feedback) and `receipt.trust_calibration` says which calibration it carries (`this deployment` or the LoCoMo `reference`); each memory's `receipt` gives its provenance (`stored_at`, `speaker`, `agent_id`, and per fact the extraction `rule` and exact `quote`). 429 above `api.reads_per_second_per_user` (a deep recall counts as 4). |
| POST | `/memory/proactive` | `message`, `agent_id?`, `limit?`, `user_id?` | Memories worth keeping in mind for this message without a question (e.g. an allergy when food is mentioned), each with `why`. |
| GET | `/memory/deletions` | `user_id?`, `limit?` | This user's deletion receipts from the tamper-evident log. `DELETE /memory/{id}` returns the receipt: what was removed, `verified_absent` per store, `all_copies_gone`, content SHA-256 (never the content), and the log hash. |
| POST | `/memory/store_batch` | `items` (1 to `api.max_batch_items`, default 100; each `{content, timestamp?, metadata?, memory_type?}`), `agent_id?`, `memory_type?`, `user_id?` | Stores several messages in one request (a session an agent framework hands over, a chat history): the same pipeline and result as storing them one by one in time order, with one embedding call for the batch (about 2× faster on the server, before the saved round trips). Returns `stored`, `skipped_duplicates` and `results`, one per item in the order of `items`. One bulk write per user at a time (**429** otherwise). Server 0.2.6+. |
| GET | `/memory/export` | `user_id?` | All of the user's memories in the open `recallix-memory` format (original messages + derived facts, topics, importance). |
| POST | `/memory/import` | `format` (`recallix` / `mem0` / `text`), `data`, `agent_id?`, `memory_type?`, `user_id?` | Imports up to 5000 memories through the normal store pipeline with their original timestamps; duplicates are skipped. Returns `stored`, `skipped_duplicates`. |
| POST | `/memory/feedback` | `recall_id`, `relevant_memory_ids[]`, `user_id?` | Labels that recall's candidates. The ranker is refitted from this automatically and a new fit is kept only if better on held-out feedback (`GET /internal/learning`, admin: attempts and outcomes; `POST /internal/learning/rollback`, `?target=answer_check` for the learned decline threshold). An empty list means none was useful (SDKs: `mem.none_used(out)`); with `learning.answer_check` on, that also teaches the answer check when to decline. SDKs: `mem.used(out)`. |
| DELETE | `/memory/{id}` | `?user_id=` | Removes the memory, its vector, long-term copy and timeline entry. 404 if not this user's. |
| GET | `/memory/timeline` | `?limit=&user_id=` | Stored memories in time order. |
| GET | `/memory/insights` | `?limit=&user_id=` | Reflections written by the replay job. |
| GET | `/memory/world-state` | `?user_id=` | Latest `user::<relation>` facts and schema shares (state-inference job). |
| GET | `/memory/prediction` | `?user_id=` | Next-topic distribution (Markov chain over the user's schema sequence). |
| GET | `/memory/plan` | `?user_id=` | Open task/calendar memories ranked by importance × recency. |
| GET | `/memory/stats` | `?user_id=` | Counts per store. |
| GET | `/memory/system/stats` | `?user_id=` | This user's index statistics and the engine's status (engine-wide counts only for admin keys). |
| GET | `/memory/profile` | `?user_id=` | World state + latest reflections. |

## Agents

| Method | Path | Body |
|---|---|---|
| POST | `/agents/store` | `content`, `agent_id`, `memory_type?`, `user_id?` |
| POST | `/agents/recall` | `query`, `agent_id`, `limit?`, `user_id?` |
| GET | `/agents/list` | this user's agents: every `agent_id` that stored memories for them, with counts (`?user_id=` for service keys) |

## Internal (admin)

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/internal/clear` | `user_id` or `"all"` | Deletes the user's data in every store and drops their vector index only. |
| POST | `/internal/reindex` | `force?`, `backfill?` | Rebuild indexes from SQLite; `backfill` first embeds memories lacking a current-model embedding. |
| POST | `/internal/jobs/run` | `job` (`state_inference`\|`prediction`\|`replay`\|`forgetting`), `user_id?` | Runs a background job now. |

## Vector engine (internal, `X-Internal-Key`)

Documented in `backend/infra_cpp/main.cpp`. Every vector operation is scoped to a
namespace (= user). Responses carry `X-Engine-Instance`; the API rebuilds indexes from
SQLite when it changes (engine restart).

## Limits

Set in `config.yaml` under `api`. Request bodies over `max_request_bytes` (1 MB; `max_import_bytes`,
10 MB, for `/memory/import` and `/memory/store_batch`) get **413**; an import of more than `max_import_items`
memories or a batch of more than `max_batch_items` gets 413, and a second import or batch for the same user while
one is running gets **429**. More than `writes_per_second_per_user`
stores or `reads_per_second_per_user` recalls per user per second get **429** (a deep recall counts as 4).
`user_id` and `agent_id` are at most 128 characters with no control characters (**422** otherwise).
