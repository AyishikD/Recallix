"""Portability: open export/import format, Mem0/text import, API endpoints, and the MCP server."""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MESSAGES = [
    "My name is Priya and I live in Pune.",
    "I'm allergic to peanuts.",
    "Project Alpha deadline is December 12.",
    "The deadline was moved to December 18.",
]


def test_export_import_roundtrip_preserves_everything_derived():
    from backend.tools import portable, replay
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        with box.silenced():
            for i, m in enumerate(MESSAGES):
                box.mm.store("alice", m, timestamp=1_700_000_000 + i, memory_type="shared")
            exported = portable.export_user(box.mm, "alice")
            json.dumps(exported)  # plain JSON
            result = portable.import_items(box.mm, "bob", portable.parse(exported, "recallic"))
            assert result == {"stored": len(MESSAGES), "skipped_duplicates": 0}
            again = portable.import_items(box.mm, "bob", portable.parse(exported, "recallic"))
            assert again["stored"] == 0  # duplicates are skipped
        snap = replay.snapshot(box.mm.episodic.db_path)
    by_user = {}
    for m in snap["memories"]:
        by_user.setdefault(m["user"], []).append((m["content"], m["timestamp"], m["facts"], m["schemas"]))
    assert by_user["alice"] == by_user["bob"]  # same messages, same timestamps -> same derived memory
    assert exported["format"] == "recallic-memory" and exported["version"] == 1


def test_import_from_mem0_and_plain_text():
    from backend.tools import portable
    mem0 = {"results": [{"memory": "Lives in Pune", "created_at": "2025-01-02T03:04:05Z"},
                        {"memory": "Is allergic to peanuts", "metadata": {"source": "chat"}}]}
    items = portable.parse(mem0, "mem0")
    assert [i["content"] for i in items] == ["Lives in Pune", "Is allergic to peanuts"]
    assert items[0]["timestamp"] == 1735787045.0 and items[1]["metadata"] == {"source": "chat"}
    text = portable.parse("- I live in Pune.\n\n• I'm allergic to peanuts.\n", "text")
    assert [i["content"] for i in text] == ["I live in Pune.", "I'm allergic to peanuts."]
    for bad, fmt in (({"format": "other"}, "recallic"), ("x", "nope")):
        try:
            portable.parse(bad, fmt)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_export_import_api(client, api_keys):
    alice, admin = api_keys["alice"], api_keys["admin"]
    for m in MESSAGES:
        assert client.post("/memory/store", json={"content": m}, headers=alice).status_code == 200
    exported = client.get("/memory/export", headers=alice).json()["data"]
    assert [m["content"] for m in exported["memories"]] == MESSAGES
    r = client.post("/memory/import", json={"format": "recallic", "data": exported, "user_id": "carol"}, headers=admin)
    assert r.status_code == 200 and r.json()["data"]["stored"] == len(MESSAGES)
    # a user key cannot import into someone else's memory
    r = client.post("/memory/import", json={"format": "text", "data": "hello", "user_id": "carol"}, headers=alice)
    assert r.status_code == 403
    assert client.post("/memory/import", json={"format": "recallic", "data": {"format": "x"}}, headers=alice).status_code == 422


# --------------------------------------------------------------------------- MCP server
class FakeMemory:
    def __init__(self):
        self.calls = []

    def recall(self, query, limit=10, agent_id="default_agent"):
        self.calls.append(("recall", query, agent_id))
        return {"status": "CONSISTENT", "recall_id": "r1", "receipt": {"reason": "answer found"},
                "memories": [{"id": 7, "content": "I'm allergic to peanuts.", "answer": "peanuts", "relevance": 0.9}]}

    def store(self, content, agent_id="default_agent", memory_type="private"):
        self.calls.append(("store", content, agent_id, memory_type))
        return {"status": "stored", "memory_id": 8}

    def feedback(self, recall_id, ids):
        return {"labelled_candidates": len(ids)}

    def delete(self, memory_id):
        return {}

    def update(self, memory_id, content):
        self.calls.append(("update", memory_id, content))
        return {"status": "edited", "memory_id": memory_id + 100, "previous_memory_id": memory_id}

    def export(self):
        return {"format": "recallic-memory", "memories": []}


def _session(server, messages):
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_mcp_handshake_tools_and_calls():
    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from recallic.mcp_server import RecallicMCP
    mem = FakeMemory()
    replies = _session(RecallicMCP(mem, agent_id="claude-desktop"), [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "recall_memory", "arguments": {"query": "What am I allergic to?"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "store_memory", "arguments": {"content": "I moved to Mumbai."}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 6, "method": "does/not/exist"},
    ])
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5, 6]  # the notification got no reply
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18" and "tools" in replies[0]["result"]["capabilities"]
    assert {t["name"] for t in replies[1]["result"]["tools"]} >= {"recall_memory", "store_memory", "confirm_answer"}
    text = replies[2]["result"]["content"][0]["text"]
    assert "status: CONSISTENT" in text and "answer: peanuts" in text and "memory 7" in text
    assert mem.calls[0] == ("recall", "What am I allergic to?", "claude-desktop")
    assert mem.calls[1] == ("store", "I moved to Mumbai.", "claude-desktop", "shared")
    assert replies[4]["error"]["code"] == -32602 and replies[5]["error"]["code"] == -32601


def test_mcp_update_memory_tool():
    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from recallic.mcp_server import RecallicMCP, TOOLS
    assert "update_memory" in {t["name"] for t in TOOLS}
    mem = FakeMemory()
    text = RecallicMCP(mem).call_tool("update_memory", {"memory_id": 7, "content": "I moved to Mumbai, not Pune."})
    assert mem.calls[-1] == ("update", 7, "I moved to Mumbai, not Pune.") and "memory 7 is now memory 107" in text


def test_mcp_unknown_protocol_version_gets_latest_supported():
    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from recallic.mcp_server import RecallicMCP, SUPPORTED_VERSIONS
    r = RecallicMCP(FakeMemory()).handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                          "params": {"protocolVersion": "1999-01-01"}})
    assert r["result"]["protocolVersion"] == SUPPORTED_VERSIONS[0]
