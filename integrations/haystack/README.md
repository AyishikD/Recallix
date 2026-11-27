# recallix-haystack: Recallix memory for Haystack agents

`RecallixMemoryStore` plugs a self-hosted [Recallix](https://github.com/AyishikD/Recallix) server into
[Haystack](https://haystack.deepset.ai) agents as their memory store. It follows Haystack's `MemoryStore`
protocol (`haystack_experimental.memory_stores`), so it drops in where the Mem0 store goes.

* **No LLM calls for memory.** Storing and searching use Recallix's local models.
* **Honest search.** When nothing answers the query, `search_memories` returns no memories instead of
  near-misses for the model to guess from.
* **Re-adding is safe.** The agent hands over the whole conversation after each run; Recallix stores each
  message once.
* **Your data stays on your server**, separated per user and per agent.

## Install

```bash
pip install "recallix-haystack[agent]"     # haystack-ai, haystack-experimental (for Agent) and the recallix SDK
```

Run Recallix ([quickstart](../../docs/quickstart.md)) and create a key with role `service` (it acts for each user):

```bash
python -m backend.tools.keys create --user my-haystack-app --role service
export RECALLIX_API_KEY=rx_...
```

## Use with a Haystack agent

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.agents import Agent
from recallix_haystack import RecallixMemoryStore

store = RecallixMemoryStore(base_url="http://localhost:8000", mode="balanced")   # or "cautious" / "helpful"
agent = Agent(chat_generator=OpenAIChatGenerator(), memory_store=store, system_prompt="You are a helpful assistant.")

agent.run(messages=[ChatMessage.from_user("I moved to Pune in March.")], memory_store_kwargs={"user_id": "alice"})
agent.run(messages=[ChatMessage.from_user("Where did I move?")], memory_store_kwargs={"user_id": "alice"})
```

Or on its own: `store.add_memories(messages=[...], user_id="alice")`,
`store.search_memories(query="Where did I move?", user_id="alice", top_k=5)`.

| Parameter | Default | What it does |
|---|---|---|
| `api_key` | `RECALLIX_API_KEY` | A Haystack `Secret`. Role `service`; `delete_all_memories` needs role `admin`. |
| `agent_id` | `haystack` | Recallix agent id; each agent reads back what it wrote. Can also be passed per call. |
| `mode` | server's `recall.mode` | `cautious` checks every answer twice; `helpful` answers more counting, ordering and "how long between" questions and adds a worked-out result first. |
| `store_roles` | `("user",)` | Which message roles are stored. The default stores what users said, not the model's replies. |

Notes:

* `search_memories` returns system `ChatMessage`s, like the Mem0 store; with `include_memory_metadata=True`
  each carries Recallix's relevance, trust and receipt. Without a `query` it returns the user's most recent
  memories.
* Haystack `filters` are not supported (Recallix scopes by user and agent) and are ignored with a warning.
* A message's original time can be given as `meta={"timestamp": <unix seconds>}`.
* Needs Recallix server 0.2.5 or newer. `add_memories` sends the messages in one request per 100 on server 0.2.6
  or newer, and one request per message before that.
* Tested with haystack-ai 2.23 + haystack-experimental 0.19 (store and Agent), and with haystack-ai 3.5 (store; the
  experimental Agent does not yet run on Haystack 3.x, which is a Haystack-side pairing, not this package). If the
  Agent import fails, pin `haystack-ai` to the release matching your `haystack-experimental`.

## Tests

Against a running Recallix: `RECALLIX_URL=... RECALLIX_SERVICE_KEY=rx_... [RECALLIX_ADMIN_KEY=rx_...] pytest tests`.
They include a real `haystack_experimental` Agent run with a scripted model.

Licence: Apache-2.0.
