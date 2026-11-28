"""
Against a running Recallic server (skipped otherwise):
    RECALLIC_URL=http://127.0.0.1:8000 RECALLIC_SERVICE_KEY=rx_... [RECALLIC_ADMIN_KEY=rx_...] pytest integrations/haystack/tests
"""
import os
import time
import uuid

import pytest

URL, KEY, ADMIN = os.environ.get("RECALLIC_URL"), os.environ.get("RECALLIC_SERVICE_KEY"), os.environ.get("RECALLIC_ADMIN_KEY")
pytestmark = pytest.mark.skipif(not (URL and KEY), reason="needs a running Recallic (RECALLIC_URL, RECALLIC_SERVICE_KEY)")


@pytest.fixture
def store():
    from haystack.utils import Secret
    from recallic_haystack import RecallicMemoryStore
    return RecallicMemoryStore(api_key=Secret.from_env_var("RECALLIC_SERVICE_KEY"), base_url=URL)


def conversation():
    from haystack.dataclasses import ChatMessage
    return [ChatMessage.from_user("I moved to Pune in March for my new job.", meta={"timestamp": time.time() - 20 * 86400}),
            ChatMessage.from_assistant("Congratulations on the move!")]


def test_follows_the_haystack_protocol_and_serializes(store):
    from recallic_haystack import RecallicMemoryStore
    for name in ("to_dict", "from_dict", "add_memories", "search_memories", "delete_all_memories", "delete_memory"):
        assert callable(getattr(store, name)), name
    d = store.to_dict()
    assert d["init_parameters"]["api_key"]["env_vars"] == ["RECALLIC_SERVICE_KEY"]  # the key itself is never written out
    again = RecallicMemoryStore.from_dict(d)
    assert again.base_url == store.base_url and again.agent_id == "haystack" and again.store_roles == ("user",)


def test_add_twice_store_once_search_finds_it(store):
    user = f"u-{uuid.uuid4().hex[:8]}"
    first = store.add_memories(messages=conversation(), user_id=user)
    again = store.add_memories(messages=conversation(), user_id=user)  # the agent re-adds the whole conversation
    assert [a["status"] for a in first] == ["stored"] and [a["status"] for a in again] == ["skipped_duplicate"]
    found = store.search_memories(query="Where did I move?", user_id=user, include_memory_metadata=True)
    assert any("Pune" in m.text for m in found) and not any("Congratulations" in m.text for m in found)
    assert all(m.role.value == "system" for m in found)
    assert "receipt" in found[0].meta["retrieved_memory_metadata"]


def test_no_record_and_separation(store):
    user = f"u-{uuid.uuid4().hex[:8]}"
    store.add_memories(messages=conversation(), user_id=user)
    assert store.search_memories(query="What is my sister's name?", user_id=user) == []
    assert store.search_memories(query="Where did I move?", user_id=f"other-{user}") == []
    assert store.search_memories(query="Where did I move?", user_id=user, agent_id="another-agent") == []
    assert "No earlier memory" in store.search_memories_as_single_message(query="What is my sister's name?", user_id=user).text


def test_recent_memories_without_a_query_and_delete_one(store):
    user = f"u-{uuid.uuid4().hex[:8]}"
    store.add_memories(messages=conversation(), user_id=user)
    recent = store.search_memories(user_id=user)
    assert any("Pune" in m.text for m in recent)
    store.delete_memory(recent[-1].meta["memory_id"], user_id=user)
    assert store.search_memories(query="Where did I move?", user_id=user) == []


def test_delete_all_needs_an_admin_key(store):
    user = f"u-{uuid.uuid4().hex[:8]}"
    store.add_memories(messages=conversation(), user_id=user)
    with pytest.raises(RuntimeError, match="admin key"):
        store.delete_all_memories(user_id=user)
    if ADMIN:
        from haystack.utils import Secret
        from recallic_haystack import RecallicMemoryStore
        RecallicMemoryStore(api_key=Secret.from_token(ADMIN), base_url=URL).delete_all_memories(user_id=user)
        assert store.search_memories(query="Where did I move?", user_id=user) == []


def test_works_inside_a_haystack_agent(store):
    """A real haystack_experimental Agent with a scripted chat generator: memories reach the model, the turn is stored."""
    from haystack import component
    from haystack.dataclasses import ChatMessage
    from haystack_experimental.components.agents import Agent

    seen = {}

    @component
    class ScriptedGenerator:
        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], tools=None, **kwargs):
            seen["messages"] = messages
            return {"replies": [ChatMessage.from_assistant("You moved to Pune.")]}

    user = f"u-{uuid.uuid4().hex[:8]}"
    store.add_memories(messages=conversation(), user_id=user)
    agent = Agent(chat_generator=ScriptedGenerator(), memory_store=store, system_prompt="You are helpful.")
    agent.warm_up()
    out = agent.run(messages=[ChatMessage.from_user("Where did I move?")], memory_store_kwargs={"user_id": user})
    assert out["last_message"].text == "You moved to Pune."
    assert any("Pune" in (m.text or "") for m in seen["messages"] if m.role.value == "system")
    recent = store.search_memories(user_id=user)
    assert any(m.text == "Where did I move?" for m in recent)  # the agent stored the user's new message
