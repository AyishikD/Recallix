"""
Different LLM clients share one memory: a Claude Desktop-style MCP client stores, a GPT-style
app using the SDK recalls (and the reverse). Real API server, real engine binary, real SDK.
"""
import io
import json
import sys

from tests.conftest import ROOT
from tests.test_clients import live_server  # noqa: F401  (pytest fixture)


def _mcp_call(server, name, arguments):
    out = io.StringIO()
    server.serve(io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": name, "arguments": arguments}}) + "\n"), out)
    return json.loads(out.getvalue())["result"]["content"][0]["text"]


def test_memory_is_shared_across_models_and_clients(live_server, api_keys):  # noqa: F811
    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from memoryos import Memory
    from memoryos.mcp_server import RecallixMCP
    key = api_keys["alice"]["X-API-Key"]

    # "Claude Desktop" stores through MCP (shared: every agent of this user may see it) ...
    claude = RecallixMCP(Memory(api_key=key, base_url=live_server), agent_id="claude-desktop")
    assert "stored" in _mcp_call(claude, "store_memory", {"content": "I'm allergic to peanuts.", "memory_type": "shared"})
    # ... and something only for itself.
    assert "stored" in _mcp_call(claude, "store_memory", {"content": "My diary password hint is blue whale.",
                                                          "memory_type": "private"})

    # A GPT-based app (any model) recalls through the SDK as a different agent.
    gpt = Memory(api_key=key, base_url=live_server)
    out = gpt.recall("What am I allergic to?", agent_id="gpt-support-bot")
    assert any("peanuts" in m["content"] for m in out["memories"])
    # The private memory stays with the agent that wrote it.
    hidden = gpt.recall("What is my diary password hint?", agent_id="gpt-support-bot")
    assert all("blue whale" not in m["content"] for m in hidden["memories"])
    assert "blue whale" in _mcp_call(claude, "recall_memory", {"query": "What is my diary password hint?"})

    # And the reverse: the GPT app stores, Claude recalls.
    gpt.store("We moved the launch to July 14.", agent_id="gpt-support-bot", memory_type="shared")
    assert "July 14" in _mcp_call(claude, "recall_memory", {"query": "When is the launch?"})

    # Another user's memory is invisible to every agent of this one.
    bob = Memory(api_key=api_keys["bob"]["X-API-Key"], base_url=live_server)
    assert all("peanuts" not in m["content"] for m in bob.recall("What am I allergic to?", agent_id="x")["memories"])
