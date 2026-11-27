"""
RecallixMemoryStore: Recallix as the memory store of a Haystack agent.

    from haystack_experimental.components.agents import Agent
    from recallix_haystack import RecallixMemoryStore

    store = RecallixMemoryStore()                      # RECALLIX_API_KEY from the environment (a "service" key)
    agent = Agent(chat_generator=..., memory_store=store)
    agent.run(messages=[ChatMessage.from_user("Where did I move?")], memory_store_kwargs={"user_id": "alice"})

It follows Haystack's MemoryStore protocol (haystack_experimental.memory_stores.types.MemoryStore), like the
Mem0 store it can replace. The agent searches before it answers and adds the conversation after; Recallix stores
each user message once (a message it already holds is skipped, so the agent re-adding the whole conversation is
safe) and, when nothing answers the query, returns no memories rather than near-misses. No LLM is called to store
or search memory.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret, deserialize_secrets_inplace
from recallix import Memory

logger = logging.getLogger(__name__)

_COMPUTED_ID = "recallix-computed"


class RecallixMemoryStore:
    """
    :param api_key: a Recallix key. Role "service" acts for each `user_id`; role "admin" is needed only for
        `delete_all_memories`.
    :param base_url: the Recallix API.
    :param agent_id: Recallix agent id for memories written and read here (each agent reads back what it wrote).
    :param mode: "cautious" | "balanced" | "helpful" for every search (None: the server's recall.mode).
    :param store_roles: roles of the messages to store; ("user",) by default, so memories are what users said.
    """

    def __init__(self, *, api_key: Secret = Secret.from_env_var("RECALLIX_API_KEY"),
                 base_url: str = "http://localhost:8000", agent_id: str = "haystack", mode: Optional[str] = None,
                 store_roles: Sequence[str] = ("user",), timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url
        self.agent_id = agent_id
        self.mode = mode
        self.store_roles = tuple(store_roles)
        self.timeout = timeout
        self.memory = Memory(api_key=api_key.resolve_value(), base_url=base_url, timeout=timeout)

    # ---------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, api_key=self.api_key.to_dict(), base_url=self.base_url, agent_id=self.agent_id,
                               mode=self.mode, store_roles=list(self.store_roles), timeout=self.timeout)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecallixMemoryStore":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    # ---------------------------------------------------------------- writing

    def add_memories(self, *, messages: list[ChatMessage], user_id: Optional[str] = None,
                     agent_id: Optional[str] = None, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Stores the text of each message whose role is in `store_roles`. Returns [{"memory_id", "memory", "status"}]
        ("stored", or "skipped_duplicate" for a message Recallix already holds). Other keyword arguments (run_id,
        ...) are accepted and ignored.
        """
        added = []
        for message in messages:
            role = message.role.value
            if role not in self.store_roles or not message.text:
                continue
            meta = message.meta or {}
            ts = meta.get("timestamp") if isinstance(meta.get("timestamp"), (int, float)) else None
            out = self.memory.store(message.text[:20000], agent_id=agent_id or self.agent_id, user_id=user_id,
                                    metadata={"speaker": "User" if role == "user" else role}, timestamp=ts)
            added.append({"memory_id": out.get("memory_id"), "memory": message.text, "status": out.get("status")})
        return added

    # ---------------------------------------------------------------- reading

    def search_memories(self, *, query: Optional[str] = None, filters: Optional[dict[str, Any]] = None, top_k: int = 5,
                        user_id: Optional[str] = None, agent_id: Optional[str] = None,
                        include_memory_metadata: bool = False, **kwargs: Any) -> list[ChatMessage]:
        """
        With a query: Recallix recall. Returns system ChatMessages (like the Mem0 store); none when Recallix finds
        nothing that answers the query. In "helpful" mode a worked-out answer (days between, or the order of
        events) comes first. Without a query: the user's most recent memories. Haystack `filters` are not
        supported by Recallix and are ignored (logged).
        """
        if filters:
            logger.warning("RecallixMemoryStore ignores Haystack filters; scope searches with user_id and agent_id")
        if not query:
            rows = self.memory.timeline(limit=top_k, user_id=user_id) or []  # the most recent, oldest first
            return [ChatMessage.from_system(text=r.get("content", ""), meta={"memory_id": str(r.get("memory_id")),
                                                                              "timestamp": r.get("timestamp")})
                    for r in rows if r.get("content")]
        out = self.memory.recall(query[:2000], limit=top_k, agent_id=agent_id or self.agent_id, user_id=user_id, mode=self.mode)
        if out.get("status") != "CONSISTENT":
            return []
        messages = []
        computed = out.get("computed_answer")
        if computed and computed.get("text"):
            meta = {"recallix_computed_answer": computed} if include_memory_metadata else {}
            messages.append(ChatMessage.from_system(
                text=f"Worked out from these memories: {computed['text']} ({out.get('receipt', {}).get('computed', '')})",
                meta={"memory_id": _COMPUTED_ID, **meta}))
        for m in (out.get("memories") or [])[:top_k]:
            meta = {"memory_id": str(m.get("id")), "timestamp": m.get("timestamp")}
            if include_memory_metadata:
                meta["retrieved_memory_metadata"] = {"relevance": m.get("relevance"), "trust": m.get("trust"),
                                                     "receipt": m.get("receipt"), "recall_id": out.get("recall_id")}
            messages.append(ChatMessage.from_system(text=m.get("content_resolved") or m.get("content", ""), meta=meta))
        return messages

    def search_memories_as_single_message(self, *, query: Optional[str] = None, filters: Optional[dict[str, Any]] = None,
                                          top_k: int = 5, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                                          **kwargs: Any) -> ChatMessage:
        found = self.search_memories(query=query, filters=filters, top_k=top_k, user_id=user_id, agent_id=agent_id)
        if not found:
            return ChatMessage.from_system(text="No earlier memory answers this.")
        return ChatMessage.from_system(text="\n".join(f"- MEMORY #{i + 1}: {m.text}" for i, m in enumerate(found)))

    # ---------------------------------------------------------------- deleting

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None, **kwargs: Any) -> None:
        if memory_id == _COMPUTED_ID:
            return
        self.memory.delete(int(memory_id), user_id=user_id)

    def delete_all_memories(self, *, user_id: Optional[str] = None, **kwargs: Any) -> None:
        """Deletes every memory of `user_id` in Recallix (all agents). Needs an admin key."""
        if not user_id:
            raise ValueError("delete_all_memories needs a user_id")
        try:
            self.memory.client.request("POST", "/internal/clear", data={"user_id": user_id})
        except Exception as e:
            raise RuntimeError(f"Recallix refused to delete all memories of {user_id!r} (an admin key is needed): {e}") from e
