"""
RecallixMemoryService: Recallix as the memory of a Google ADK agent.

    from recallix_adk import RecallixMemoryService
    memory = RecallixMemoryService(api_key="rx_...")          # a Recallix key with role "service"
    runner = Runner(agent=agent, app_name="my_app", session_service=..., memory_service=memory)

ADK hands over sessions (add_session_to_memory) or the latest turn (add_events_to_memory); this stores each text
event in Recallix with its original time. search_memory recalls with Recallix and returns ADK MemoryEntry items.
When Recallix finds nothing that answers the query it returns no memories, so the agent is not handed
near-misses to guess from. No LLM is called to store or search memory.

Scope: Recallix user = the ADK user_id; Recallix agent = the ADK app_name, so each app reads back what it wrote.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from recallix import Memory

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session


def _event_text(event: "Event") -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


def _iso(ts: Any) -> Optional[str]:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


class RecallixMemoryService(BaseMemoryService):
    """
    api_key: a Recallix key with role "service" (it acts for each ADK user_id).
    base_url: the Recallix API.
    mode: "cautious" | "balanced" | "helpful" for every search (None: the server's recall.mode).
    limit: memories per search.
    store_authors: which event authors to store: ("user",) by default, so memories are what users said. Pass None
        to store every author, including the agent's own replies.
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", *, mode: Optional[str] = None,
                 limit: int = 10, store_authors: Optional[Sequence[str]] = ("user",), timeout: float = 30.0,
                 client: Optional[Memory] = None):
        self.memory = client or Memory(api_key=api_key, base_url=base_url, timeout=timeout)
        self.mode = mode
        self.limit = limit
        self.store_authors = tuple(store_authors) if store_authors is not None else None

    # ---------------------------------------------------------------- writing

    def _store_events(self, app_name: str, user_id: str, events: Sequence["Event"],
                      custom_metadata: Optional[Mapping[str, object]] = None) -> None:
        for event in events:
            author = getattr(event, "author", None) or "user"
            if self.store_authors is not None and author not in self.store_authors:
                continue
            text = _event_text(event)
            if not text:
                continue
            metadata = {"speaker": "User" if author == "user" else author}
            if custom_metadata:
                metadata["adk"] = dict(custom_metadata)
            # Recallix skips a message it already holds for this user and agent, so re-adding a session is safe
            self.memory.store(text[:20000], agent_id=app_name, user_id=user_id, metadata=metadata,
                              timestamp=getattr(event, "timestamp", None))

    async def add_session_to_memory(self, session: "Session") -> None:
        await asyncio.to_thread(self._store_events, session.app_name, session.user_id, list(session.events or []))

    async def add_events_to_memory(self, *, app_name: str, user_id: str, events: Sequence["Event"],
                                   session_id: Optional[str] = None,
                                   custom_metadata: Optional[Mapping[str, object]] = None) -> None:
        await asyncio.to_thread(self._store_events, app_name, user_id, list(events), custom_metadata)

    async def add_memory(self, *, app_name: str, user_id: str, memories: Sequence[MemoryEntry],
                         custom_metadata: Optional[Mapping[str, object]] = None) -> None:
        def write():
            for entry in memories:
                text = "\n".join(p.text for p in (entry.content.parts or []) if getattr(p, "text", None)).strip()
                if not text:
                    continue
                ts = None
                if entry.timestamp:
                    try:
                        ts = datetime.fromisoformat(entry.timestamp).timestamp()
                    except ValueError:
                        ts = None
                self.memory.store(text[:20000], agent_id=app_name, user_id=user_id, timestamp=ts,
                                  metadata={"speaker": entry.author or "User", **({"adk": dict(custom_metadata)} if custom_metadata else {})})
        await asyncio.to_thread(write)

    # ---------------------------------------------------------------- reading

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        out = await asyncio.to_thread(lambda: self.memory.recall(query[:2000], limit=self.limit, agent_id=app_name,
                                                                 user_id=user_id, mode=self.mode))
        if out.get("status") != "CONSISTENT":
            return SearchMemoryResponse(memories=[])  # Recallix found nothing that answers it: no near-misses
        entries = []
        computed = out.get("computed_answer")
        if computed and computed.get("text"):
            entries.append(MemoryEntry(
                id="recallix-computed", author="recallix", timestamp=None,
                content=types.Content(role="user", parts=[types.Part(text=f"Worked out from the memories below: {computed['text']}"
                                                                         f" ({out.get('receipt', {}).get('computed', '')})")]),
                custom_metadata={"computed_answer": computed}))
        for m in (out.get("memories") or [])[: self.limit]:
            speaker = ((m.get("metadata") or {}).get("client") or {}).get("speaker") if isinstance(m.get("metadata"), dict) else None
            entries.append(MemoryEntry(
                id=str(m.get("id")),
                author="user" if speaker in (None, "User") else speaker,
                timestamp=_iso(m.get("timestamp")),
                content=types.Content(role="user", parts=[types.Part(text=m.get("content_resolved") or m.get("content", ""))]),
                custom_metadata={"relevance": m.get("relevance"), "trust": m.get("trust"),
                                 "receipt": m.get("receipt"), "recallix_recall_id": out.get("recall_id")}))
        return SearchMemoryResponse(memories=entries)

    async def delete_memory(self, *, app_name: str, user_id: str, memory_id: Optional[str] = None) -> None:
        if memory_id is None or memory_id == "recallix-computed":
            return
        await asyncio.to_thread(self.memory.delete, int(memory_id), user_id)
