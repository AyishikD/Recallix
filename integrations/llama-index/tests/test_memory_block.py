"""
Live tests against a running Recallic:
    RECALLIC_URL=http://127.0.0.1:8000 RECALLIC_SERVICE_KEY=rx_... pytest tests
"""
import asyncio
import os
import uuid
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, CompletionResponse, LLMMetadata
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.memory import Memory
from llama_index.memory.recallic import RecallicMemoryBlock

URL, KEY = os.environ.get("RECALLIC_URL"), os.environ.get("RECALLIC_SERVICE_KEY")
pytestmark = pytest.mark.skipif(not (URL and KEY), reason="needs RECALLIC_URL and RECALLIC_SERVICE_KEY")


def _block(user, **kw):
    return RecallicMemoryBlock(api_key=KEY, base_url=URL, user_id=user, agent_id="li-test", timeout=120, **kw)


def _memory(user, **kw):
    return Memory.from_defaults(session_id=f"{user}-{uuid.uuid4().hex[:6]}", memory_blocks=[_block(user, **kw)])


def _system_text(messages):
    return "\n".join(m.content or "" for m in messages if m.role.value == "system")


def test_recalls_across_sessions_and_declines_honestly():
    user = f"li-{uuid.uuid4().hex[:8]}"

    async def run():
        first = _memory(user)
        await first.aput(ChatMessage(role="user", content="I moved to Pune in March, and my sister Mira is a pediatrician in Goa."))
        await first.aget(input="Thanks!")  # a turn: recall, then store the new user message
        second = _memory(user)  # a brand-new session: nothing in its short-term memory
        found = await second.aget(input="Which city did I move to?")
        never = await second.aget(input="What is my brother's name?")
        return found, never

    found, never = asyncio.run(run())
    assert "Pune" in _system_text(found)
    text = _system_text(never)
    assert "don't have a record" in text and "Pune" not in text


def test_the_question_is_not_recalled_as_its_own_answer():
    user = f"li-{uuid.uuid4().hex[:8]}"

    async def run():
        mem = _memory(user)
        await mem.aput(ChatMessage(role="user", content="What is my brother's name?"))
        await mem.aget()  # stores the question (after recalling)
        return await _memory(user).aget(input="What is my brother's name?")

    assert "don't have a record" in _system_text(asyncio.run(run()))


def test_no_decline_note_when_turned_off():
    user = f"li-{uuid.uuid4().hex[:8]}"
    out = asyncio.run(_memory(user, tell_model_on_decline=False).aget(input="What is my brother's name?"))
    assert "Recallic" not in _system_text(out) and "record" not in _system_text(out)


class RecordingLLM(CustomLLM):
    """A scripted model that records the prompt it is given."""
    seen: list = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=False)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.seen.append(prompt)
        return CompletionResponse(text="Noted.")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        yield self.complete(prompt)


def test_inside_a_llamaindex_chat_engine():
    from llama_index.core.chat_engine import SimpleChatEngine
    user = f"li-{uuid.uuid4().hex[:8]}"
    llm = RecordingLLM()
    SimpleChatEngine.from_defaults(llm=llm, memory=_memory(user)).chat("My cat is called Biscuit.")
    SimpleChatEngine.from_defaults(llm=llm, memory=_memory(user)).chat("What is my cat called?")
    assert "Biscuit" in llm.seen[-1] and "Things this user told you before" in llm.seen[-1]
