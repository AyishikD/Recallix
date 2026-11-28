# recallic (Python)

Client and MCP server for [Recallic](https://github.com/Recallic/Recallic): an LLM-free memory
layer for AI agents. Recallic runs on your own machine, needs no LLM calls, and declines
instead of guessing.

```bash
pip install recallic
```

You need a running Recallic server. The quickest way is
`docker compose up -d` in the [repository](https://github.com/Recallic/Recallic); see the quickstart there.

```python
from recallic import Memory

mem = Memory(api_key="rx_...")              # base_url defaults to http://localhost:8000
mem.store("I moved to Pune last month for a job at Infosys.")
out = mem.recall("Where do I live?")
print(out["status"], [m["content"] for m in out["memories"]])   # CONSISTENT ['I moved to Pune ...']
print(mem.recall("What is my brother's name?")["status"])        # ABSTAINED: nothing answers it
mem.used(out)                               # one line of feedback: Recallic learns from it on its own
```

How readily recall answers is one option (server 0.2.5 or newer; default `"balanced"`):

```python
mem.recall("Which came first, the Goa trip or the Pune workshop?", mode="helpful")   # fewer refusals, computed_answer
mem.recall("What dose did the doctor prescribe?", mode="cautious")                     # checks every answer twice
```

Storing a whole conversation at once (server 0.2.6 or newer; an older server gets one store per message):

```python
mem.store_many(["I moved to Pune in March.", {"content": "Mira got married.", "timestamp": 1717200000}])
```

Correcting a memory keeps when it was said; the memory gets a new id (server 0.2.7 or newer):

```python
mem.update(memory_id, "I moved to Mumbai in March.")
```

## Changes

* **0.3.0**: renamed from `recallix` to **`recallic`**: `pip install recallic`, `from recallic import Memory`, `recallic-mcp`.
  Nothing else changed; `RECALLIX_API_KEY` / `RECALLIX_URL` and the `recallix-mcp` command still work.
* **0.2.7**: `update(memory_id, content)` corrects a memory (`PUT /memory/{id}`); the MCP server has an `update_memory` tool.
* **0.2.6**: `store_many(items)` stores many messages in one request per 100 (`POST /memory/store_batch`).
* **0.2.5**: `recall(..., mode="cautious" | "balanced" | "helpful")`; `store(..., timestamp=...)` keeps when a message was originally said.
* **0.2.4**: `requests>=2.33.0`.

## MCP (Claude Desktop, Cursor, any MCP client)

```json
"mcpServers": {
  "recallic": {
    "command": "recallic-mcp",
    "env": {"RECALLIC_API_KEY": "rx_...", "RECALLIC_AGENT_ID": "claude-desktop"}
  }
}
```

Environment: `RECALLIC_API_KEY` (required), `RECALLIC_URL` (default `http://localhost:8000`),
`RECALLIC_AGENT_ID` (default `mcp`).

License: Apache-2.0.
