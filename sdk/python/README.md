# recallix (Python)

Client and MCP server for [Recallix](https://github.com/AyishikD/Recallix): an LLM-free memory
layer for AI agents. Recallix runs on your own machine, needs no LLM calls, and declines
instead of guessing.

```bash
pip install recallix
```

You need a running Recallix server. The quickest way is
`docker compose up -d` in the [repository](https://github.com/AyishikD/Recallix); see the quickstart there.

```python
from recallix import Memory

mem = Memory(api_key="rx_...")              # base_url defaults to http://localhost:8000
mem.store("I moved to Pune last month for a job at Infosys.")
out = mem.recall("Where do I live?")
print(out["status"], [m["content"] for m in out["memories"]])   # CONSISTENT ['I moved to Pune ...']
print(mem.recall("What is my brother's name?")["status"])        # ABSTAINED: nothing answers it
```

## MCP (Claude Desktop, Cursor, any MCP client)

```json
"mcpServers": {
  "recallix": {
    "command": "recallix-mcp",
    "env": {"RECALLIX_API_KEY": "rx_...", "RECALLIX_AGENT_ID": "claude-desktop"}
  }
}
```

Environment: `RECALLIX_API_KEY` (required), `RECALLIX_URL` (default `http://localhost:8000`),
`RECALLIX_AGENT_ID` (default `mcp`).

License: Apache-2.0.
