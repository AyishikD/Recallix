# API reference

All endpoints require `X-API-Key`. Responses are `{"status": "success", "data": ..., "timestamp": ...}`;
errors are HTTP 4xx/5xx with `{"detail": ...}`.

**Keys and users.** A `user` key acts only as its own user. `service` and `admin` keys may pass
`user_id` (body for POST, query string for GET/DELETE) to act for an end user; a `user` key
that does so gets 403. `/internal/*` needs an `admin` key.

**Agents.** `agent_id` defaults to `default_agent`. A memory is visible to an agent if it is
`shared` or that agent's own `private` memory.

## Memory

| Method | Path | Body / query | Notes |
|---|---|---|---|
| POST | `/memory/store` | `content`, `agent_id?`, `memory_type? ("private"\|"shared")`, `metadata?`, `user_id?` | Returns `memory_id`, `importance`, `schemas`, `status` (`stored` / `skipped_duplicate`). 429 above `api.writes_per_second_per_user`. |
| POST | `/memory/recall` | `query`, `limit? (1-100)`, `agent_id?`, `user_id?` | Returns `recall_id`, `status` (`CONSISTENT` / `ABSTAINED` / `NO_MATCH`), `memories[]` each with `relevance` and an `explain` feature breakdown. |
| POST | `/memory/feedback` | `recall_id`, `relevant_memory_ids[]`, `user_id?` | Labels that recall's candidates for ranker training. |
| DELETE | `/memory/{id}` | `?user_id=` | Removes the memory, its vector, long-term copy and timeline entry. 404 if not this user's. |
| GET | `/memory/timeline` | `?limit=&user_id=` | Stored memories in time order. |
| GET | `/memory/insights` | `?limit=&user_id=` | Reflections written by the replay job. |
| GET | `/memory/world-state` | `?user_id=` | Latest `user::<relation>` facts and schema shares (state-inference job). |
| GET | `/memory/prediction` | `?user_id=` | Next-topic distribution (Markov chain over the user's schema sequence). |
| GET | `/memory/plan` | `?user_id=` | Open task/calendar memories ranked by importance × recency. |
| GET | `/memory/stats` | `?user_id=` | Counts per store. |
| GET | `/memory/system/stats` | `?user_id=` | Engine status and this user's index statistics. |
| GET | `/memory/profile` | `?user_id=` | World state + latest reflections. |

## Agents

| Method | Path | Body |
|---|---|---|
| POST | `/agents/store` | `content`, `agent_id`, `memory_type?`, `user_id?` |
| POST | `/agents/recall` | `query`, `agent_id`, `limit?`, `user_id?` |
| GET | `/agents/list` | registered agents |

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
