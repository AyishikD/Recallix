"""
Recallix as an MCP server: give Claude Desktop, Cursor, or any Model Context Protocol
client a memory that you own and that works with every model.

    RECALLIX_API_KEY=rx_... python -m recallix.mcp_server

Environment:
    RECALLIX_API_KEY    your Recallix API key (required)
    RECALLIX_URL        API base URL (default http://localhost:8000)
    RECALLIX_AGENT_ID   agent name for memories written through this client (default "mcp")

Claude Desktop / Cursor configuration (mcpServers):

    "recallix": {
      "command": "python", "args": ["-m", "recallix.mcp_server"],
      "env": {"RECALLIX_API_KEY": "rx_...", "RECALLIX_AGENT_ID": "claude-desktop"}
    }

Memories written by one client with memory_type "shared" are visible to every client and
agent of the same user; "private" ones only to the agent that wrote them.

Implementation: JSON-RPC 2.0 over stdio, one message per line (the MCP stdio transport);
no dependencies beyond the SDK. Logs go to stderr — stdout carries only protocol messages.
"""
import json
import os
import sys

SERVER_INFO = {"name": "recallix", "version": "1.0.0"}
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "Recallix is the user's long-term memory. Call recall_memory before answering questions that may "
    "depend on what the user told you earlier; if its status is ABSTAINED or NO_MATCH, say you don't know "
    "rather than guessing. Call store_memory when the user shares a durable fact, preference or decision. "
    "When a recalled memory was what answered the user, call confirm_answer. When the user corrects something "
    "you recalled, call update_memory on that memory instead of storing a second, contradicting one."
)

TOOLS = [
    {"name": "recall_memory",
     "description": "Search the user's memory. Returns status (CONSISTENT = an answer was found, ABSTAINED = "
                    "nothing answers it, NO_MATCH = nothing related), the matching memories with the extracted "
                    "answer, and the reason for the decision.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "The question, in natural language."},
         "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
         "deep": {"type": "boolean", "default": False,
                  "description": "Read the candidates more carefully (slower). Use when the answer is spread over "
                                 "several things the user said, e.g. 'what are all the places I've visited?'."}},
         "required": ["query"]}},
    {"name": "store_memory",
     "description": "Remember something the user said: a fact, preference, plan or decision. Write it as the "
                    "user would say it, e.g. \"I'm allergic to peanuts.\"",
     "inputSchema": {"type": "object", "properties": {
         "content": {"type": "string"},
         "memory_type": {"type": "string", "enum": ["shared", "private"], "default": "shared",
                         "description": "shared: visible to all of the user's agents; private: only this one."}},
         "required": ["content"]}},
    {"name": "confirm_answer",
     "description": "Tell Recallix which recalled memories actually answered the question (improves ranking and "
                    "makes similar future questions find them).",
     "inputSchema": {"type": "object", "properties": {
         "recall_id": {"type": "string"},
         "memory_ids": {"type": "array", "items": {"type": "integer"}}},
         "required": ["recall_id", "memory_ids"]}},
    {"name": "relevant_memories",
     "description": "Given what the user just said, return memories worth keeping in mind even though they did not "
                    "ask (e.g. an allergy when they mention a restaurant), each with the reason.",
     "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    {"name": "update_memory",
     "description": "Correct a memory the user says is wrong or out of date (e.g. 'it was Mumbai, not Pune'): find "
                    "it with recall_memory, then replace its text. It keeps the date it was originally said; the old "
                    "version is deleted.",
     "inputSchema": {"type": "object", "properties": {
         "memory_id": {"type": "integer"},
         "content": {"type": "string", "description": "The corrected memory, written as the user would say it."}},
         "required": ["memory_id", "content"]}},
    {"name": "forget_memory",
     "description": "Delete a memory permanently (its vector, long-term copy and timeline entry too).",
     "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "integer"}},
                     "required": ["memory_id"]}},
    {"name": "export_memories",
     "description": "Export all of the user's memories in the open recallix-memory format.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _log(*args):
    print(*args, file=sys.stderr, flush=True)


class RecallixMCP:
    def __init__(self, memory, agent_id="mcp"):
        self.memory, self.agent_id = memory, agent_id

    # -- tools ------------------------------------------------------------
    def _recall(self, query, limit=5, deep=False):
        out = self.memory.recall(query, limit=int(limit), agent_id=self.agent_id, **({"deep": True} if deep else {}))
        lines = [f"status: {out.get('status')}", f"reason: {(out.get('receipt') or {}).get('reason', '')}",
                 f"recall_id: {out.get('recall_id')}"]
        for m in out.get("memories", []):
            answer = f" | answer: {m['answer']}" if m.get("answer") else ""
            lines.append(f"- [memory {m.get('id')}] {m.get('content')}{answer} (relevance {m.get('relevance')})")
        return "\n".join(lines)

    def _store(self, content, memory_type="shared"):
        out = self.memory.store(content, agent_id=self.agent_id, memory_type=memory_type)
        return f"{out.get('status')}: memory {out.get('memory_id')}"

    def _confirm(self, recall_id, memory_ids):
        out = self.memory.feedback(recall_id, [int(i) for i in memory_ids])
        return f"confirmed ({out.get('labelled_candidates')} candidates labelled)"

    def _relevant(self, message):
        rows = self.memory.proactive(message, agent_id=self.agent_id)
        return "\n".join(f"- [memory {r['memory_id']}] {r['content']} ({r['why']})" for r in rows) or "nothing relevant"

    def _update(self, memory_id, content):
        out = self.memory.update(int(memory_id), content)
        return f"{out.get('status')}: memory {memory_id} is now memory {out.get('memory_id')}"

    def _forget(self, memory_id):
        self.memory.delete(int(memory_id))
        return f"memory {memory_id} deleted"

    def _export(self):
        return json.dumps(self.memory.export(), ensure_ascii=False)

    def call_tool(self, name, args):
        handlers = {"recall_memory": self._recall, "store_memory": self._store, "confirm_answer": self._confirm,
                    "forget_memory": self._forget, "update_memory": self._update, "relevant_memories": self._relevant, "export_memories": self._export}
        if name not in handlers:
            raise KeyError(name)
        return handlers[name](**(args or {}))

    # -- protocol ---------------------------------------------------------
    def handle(self, msg):
        """Response dict for a request, or None for a notification."""
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if mid is None:  # notification (e.g. notifications/initialized): no reply
            return None
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
                result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                try:
                    text = self.call_tool(params.get("name"), params.get("arguments"))
                    result = {"content": [{"type": "text", "text": text}], "isError": False}
                except KeyError:
                    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"}}
                except Exception as e:  # tool failures are reported to the model, not as protocol errors
                    result = {"content": [{"type": "text", "text": f"Recallix error: {e}"}], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, stdin=sys.stdin, stdout=sys.stdout):
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}}) + "\n")
                stdout.flush()
                continue
            reply = self.handle(msg)
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def main():
    from .memory import Memory
    key = os.environ.get("RECALLIX_API_KEY")
    if not key:
        _log("RECALLIX_API_KEY is not set")
        return 1
    memory = Memory(api_key=key, base_url=os.environ.get("RECALLIX_URL", "http://localhost:8000"))
    _log(f"recallix MCP server ready ({os.environ.get('RECALLIX_URL', 'http://localhost:8000')})")
    RecallixMCP(memory, os.environ.get("RECALLIX_AGENT_ID", "mcp")).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
