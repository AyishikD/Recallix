"""
RecallicMemoryBlock: Recallic as long-term memory of a LlamaIndex agent.

    from llama_index.core.memory import Memory
    from llama_index.memory.recallic import RecallicMemoryBlock

    block = RecallicMemoryBlock(api_key="rx_...", user_id="alice")       # a "service" key acts for each user
    memory = Memory.from_defaults(session_id="alice-1", memory_blocks=[block])
    agent.run("Where did I move?", memory=memory)

Each turn, the block recalls what this user said before that bears on their latest message (in any earlier
session or agent) and gives it to the model; when nothing answers it, it says so instead of handing over
near-misses. Then it stores the user's new messages (re-sending one Recallic already holds is skipped). No LLM is
called to store or search memory.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, TextBlock
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.memory.memory import BaseMemoryBlock
from recallic import Memory as RecallicClient

DECLINE_NOTE = ("None of this user's earlier messages answers their latest message. If it depends on something "
                "they would have told you before, say you don't have a record of it rather than guess.")


def _text(message: ChatMessage) -> str:
    return " ".join(b.text for b in message.blocks if isinstance(b, TextBlock) and b.text).strip()


def _when(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return ""


class RecallicMemoryBlock(BaseMemoryBlock[str]):
    """
    A LlamaIndex memory block backed by a self-hosted Recallic server.

    api_key: a Recallic key. Role "service" acts for `user_id`; a "user" key acts for itself (leave user_id None).
    user_id: whose memory this is (Recallic keeps each user's memories apart).
    agent_id: Recallic agent id; each agent reads back what it wrote (and what any agent shared).
    mode: "cautious" | "balanced" | "helpful" (None: the server's recall.mode).
    store_roles: which message roles are stored ("user" by default: memories are what the user said).
    store_immediately: store new messages every turn (after recalling), not only when LlamaIndex moves them out of
        short-term memory, so other sessions and agents see them at once.
    tell_model_on_decline: when Recallic has no answering memory, tell the model not to guess about the user's past.
    """

    name: str = Field(default="recallic_memory", description="The name of the memory block.")
    description: Optional[str] = Field(default="What this user told you in earlier conversations (Recallic).")
    api_key: str = Field(exclude=True, repr=False, description="Recallic API key.")
    base_url: str = Field(default="http://localhost:8000")
    user_id: Optional[str] = Field(default=None)
    agent_id: str = Field(default="llamaindex")
    mode: Optional[str] = Field(default=None)
    limit: int = Field(default=5, ge=1, le=20)
    store_roles: Tuple[str, ...] = Field(default=("user",))
    store_immediately: bool = Field(default=True)
    tell_model_on_decline: bool = Field(default=True)
    timeout: float = Field(default=30.0)

    _client: RecallicClient = PrivateAttr()
    _sent: set = PrivateAttr(default_factory=set)

    def __init__(self, **data: Any) -> None:
        client = data.pop("client", None)
        super().__init__(**data)
        self._client = client or RecallicClient(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    # ---------------------------------------------------------------- reading

    def _recall(self, query: str) -> str:
        out = self._client.recall(query[:2000], limit=self.limit, agent_id=self.agent_id, user_id=self.user_id,
                                  mode=self.mode)
        memories = out.get("memories") or []
        if out.get("status") != "CONSISTENT" or not memories:
            return DECLINE_NOTE if self.tell_model_on_decline else ""
        lines = ["Things this user told you before (newest facts win; use only what bears on their message):"]
        for m in memories[: self.limit]:
            when = _when(m.get("timestamp"))
            lines.append(f"- {('[' + when + '] ') if when else ''}{m.get('content_resolved') or m.get('content', '')}")
        computed = out.get("computed_answer")
        if computed and computed.get("text"):
            lines.append(f"Worked out from these memories: {computed['text']} "
                         f"({(out.get('receipt') or {}).get('computed', '')}).")
        return "\n".join(lines)

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        messages = messages or []
        last_user = next((m for m in reversed(messages) if m.role.value == "user" and _text(m)), None)
        content = await asyncio.to_thread(self._recall, _text(last_user)) if last_user else ""
        if self.store_immediately:  # after recalling, so the question does not recall itself
            await self._store(messages)
        return content

    # ---------------------------------------------------------------- writing

    def _new_items(self, messages: List[ChatMessage]) -> List[dict]:
        items = []
        for m in messages:
            role, text = m.role.value, _text(m)
            if role not in self.store_roles or not text:
                continue
            key = (role, text)
            if key in self._sent:
                continue
            self._sent.add(key)
            ts = m.additional_kwargs.get("timestamp")
            items.append({"content": text[:20000], "timestamp": ts if isinstance(ts, (int, float)) else None,
                          "metadata": {"speaker": "User" if role == "user" else role,
                                       **({"session_id": m.additional_kwargs["session_id"]}
                                          if m.additional_kwargs.get("session_id") else {})}})
        return items

    async def _store(self, messages: List[ChatMessage]) -> None:
        items = self._new_items(messages)
        if items:
            await asyncio.to_thread(self._client.store_many, items, agent_id=self.agent_id, user_id=self.user_id)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Messages LlamaIndex moves out of short-term memory (already stored when store_immediately is on)."""
        await self._store(messages)
