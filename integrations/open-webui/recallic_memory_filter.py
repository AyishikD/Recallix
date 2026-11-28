"""
title: Recallic Memory
author: Recallic
author_url: https://github.com/Recallic/Recallic
version: 0.1.1
license: Apache-2.0
required_open_webui_version: 0.6.0
description: Long-term memory for every chat, from a self-hosted Recallic server. No LLM calls for memory: it recalls what each user told any model before, says when it has no record instead of guessing, and stores new messages. Your data stays on your server.
"""
# Open WebUI filter (Admin Panel > Functions > Import from URL, or paste this file). For each chat turn:
#   1. recall what this user said before that bears on their new message (POST /memory/recall),
#   2. give the model those memories as a short system note, with the sources shown in the chat,
#      and, when Recallic has no record, tell the model so instead of letting it guess,
#   3. store the user's new message (POST /memory/store), after recalling so a question does not recall itself.
# One Recallic "service" key serves every Open WebUI user; each user's memories are kept apart by their user id.
# Any Recallic failure is reported as a status line and the chat goes on without memory.

import asyncio
from datetime import datetime, timezone
from typing import Literal, Optional

import requests
from pydantic import BaseModel, Field

MARKER = "### Recallic memory"


def _text(content):
    """A message's text: a plain string, or the text parts of a multimodal list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _uv(user_valves, key, default):
    """A per-user setting, whether Open WebUI passes UserValves as an object or as a dict."""
    if user_valves is None:
        return default
    if isinstance(user_valves, dict):
        return user_valves.get(key, default)
    return getattr(user_valves, key, default)


def _when(memory):
    ts = memory.get("timestamp")
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return ""


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Lower runs first among filters.")
        recallic_url: str = Field(default="http://localhost:8000",
                                  description="Recallic API. From an Open WebUI container to a host install: http://host.docker.internal:8000")
        api_key: str = Field(default="", description="A Recallic key with role 'service': python -m backend.tools.keys create --user open-webui --role service")
        agent_id: str = Field(default="open-webui", description="Recallic agent id for memories written and read here.")
        mode: Literal["cautious", "balanced", "helpful"] = Field(
            default="balanced", description="How readily recall answers: cautious checks every answer twice; helpful answers more counting, ordering and time questions.")
        limit: int = Field(default=5, ge=1, le=20, description="Memories given to the model per turn.")
        store_user_messages: bool = Field(default=True, description="Store each new user message in Recallic.")
        store_assistant_replies: bool = Field(default=False, description="Also store the model's replies (off: memories are what the user said).")
        tell_model_on_decline: bool = Field(default=True, description="When Recallic has no answering memory, tell the model not to guess about the user's past.")
        show_sources: bool = Field(default=True, description="Show recalled memories as sources under the reply.")
        timeout_seconds: float = Field(default=8.0, description="Per request. The first request after Recallic starts can take longer while its models load.")

    class UserValves(BaseModel):
        enabled: bool = Field(default=True, description="Use Recallic memory in my chats.")
        mode: Literal["", "cautious", "balanced", "helpful"] = Field(default="", description="Overrides the admin's mode for me (empty: admin's).")

    def __init__(self):
        self.valves = self.Valves()

    # ---------------------------------------------------------------- Recallic calls

    def _post(self, path, payload):
        r = requests.post(self.valves.recallic_url.rstrip("/") + path, json=payload,
                          headers={"X-API-Key": self.valves.api_key}, timeout=self.valves.timeout_seconds)
        r.raise_for_status()
        return r.json().get("data", {})

    async def _call(self, path, payload):
        return await asyncio.to_thread(self._post, path, payload)

    # ---------------------------------------------------------------- the note the model sees

    def _note(self, out):
        status = out.get("status")
        memories = out.get("memories") or []
        if status == "CONSISTENT" and memories:
            lines = [MARKER,
                     "Things this user told you in earlier chats (newest facts win; use only what bears on their message, "
                     "and do not mention this note):"]
            for m in memories[: self.valves.limit]:
                when = _when(m)
                lines.append(f"- {('[' + when + '] ') if when else ''}{m.get('content_resolved') or m.get('content', '')}")
            computed = out.get("computed_answer")
            if computed and computed.get("text"):
                lines.append(f"Worked out from these memories: {computed['text']} ({out.get('receipt', {}).get('computed', '')}).")
            return "\n".join(lines)
        if status in ("ABSTAINED", "NO_MATCH") and self.valves.tell_model_on_decline:
            return (f"{MARKER}\nNone of this user's earlier messages answers what they are asking now. If it depends on "
                    "something they would have told you before, say you don't have a record of it rather than guess.")
        return None

    @staticmethod
    def _add_system_note(body, note):
        messages = body.setdefault("messages", [])
        for msg in messages:  # one note per turn: replace any earlier one (idempotent)
            if msg.get("role") == "system" and MARKER in _text(msg.get("content")):
                msg["content"] = note
                return
        messages.insert(0, {"role": "system", "content": note})

    # ---------------------------------------------------------------- filter hooks

    async def inlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None) -> dict:
        user = __user__ or {}
        user_valves = user.get("valves")
        if not _uv(user_valves, "enabled", True):
            return body
        if not self.valves.api_key or not user.get("id"):
            return body
        last = next((m for m in reversed(body.get("messages") or []) if m.get("role") == "user"), None)
        text = _text(last.get("content")).strip() if last else ""
        if not text:
            return body
        mode = _uv(user_valves, "mode", "") or self.valves.mode

        async def status(description, done=True):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": description, "done": done, "hidden": False}})

        try:
            out = await self._call("/memory/recall", {"query": text[:2000], "user_id": user["id"], "agent_id": self.valves.agent_id,
                                                       "limit": self.valves.limit, "mode": mode})
        except Exception as e:  # Recallic down, wrong key, timeout: the chat goes on without memory
            await status(f"Recallic memory unavailable ({type(e).__name__})")
            out = None

        if out is not None:
            note = self._note(out)
            if note:
                self._add_system_note(body, note)
            memories = out.get("memories") or []
            if out.get("status") == "CONSISTENT" and memories:
                await status(f"Recalled {min(len(memories), self.valves.limit)} memories ({mode})")
                if self.valves.show_sources and __event_emitter__:
                    for m in memories[: self.valves.limit]:
                        await __event_emitter__({"type": "source", "data": {
                            "source": {"name": f"Recallic memory {m.get('id')}", "id": str(m.get("id"))},
                            "document": [m.get("content_resolved") or m.get("content", "")],
                            "metadata": [{"source": "Recallic", "name": _when(m) or "memory"}]}})
            else:
                await status("No memory answers this (Recallic)")

        if self.valves.store_user_messages:
            try:
                await self._call("/memory/store", {"content": text[:20000], "user_id": user["id"], "agent_id": self.valves.agent_id})
            except Exception:
                pass  # not storing one message must never break the chat
        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.store_assistant_replies or not self.valves.api_key:
            return body
        user = __user__ or {}
        user_valves = user.get("valves")
        if not user.get("id") or not _uv(user_valves, "enabled", True):
            return body
        reply = next((m for m in reversed(body.get("messages") or []) if m.get("role") == "assistant"), None)
        text = _text(reply.get("content")).strip() if reply else ""
        if text:
            try:
                await self._call("/memory/store", {"content": text[:20000], "user_id": user["id"], "agent_id": self.valves.agent_id,
                                                   "metadata": {"speaker": "Assistant"}})
            except Exception:
                pass
        return body
