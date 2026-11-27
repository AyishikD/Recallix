"""The Open WebUI filter (integrations/open-webui), run as Open WebUI runs it, against the real Recallix API."""
import asyncio
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_filter():
    spec = importlib.util.spec_from_file_location("recallix_memory_filter",
                                                  ROOT / "integrations/open-webui/recallix_memory_filter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Response:
    def __init__(self, r):
        self._r = r

    def raise_for_status(self):
        if self._r.status_code >= 400:
            raise RuntimeError(f"HTTP {self._r.status_code}")

    def json(self):
        return self._r.json()


@pytest.fixture
def owui(client, monkeypatch):
    """The filter module, with its HTTP calls answered by the test Recallix server, and a service key."""
    from backend.api import keys
    mod = load_filter()
    monkeypatch.setattr(mod.requests, "post",
                        lambda url, json, headers, timeout: Response(client.post(url.split("8000", 1)[1], json=json, headers=headers)))
    f = mod.Filter()
    f.valves.api_key = keys.create("open-webui", "service")
    return mod, f


def turn(f, user_id, text, history=(), user_valves=None):
    events = []

    async def emit(e):
        events.append(e)
    body = {"model": "m", "messages": [*history, {"role": "user", "content": text}]}
    user = {"id": user_id, "name": "N", **({"valves": user_valves} if user_valves is not None else {})}
    out = asyncio.run(f.inlet(body, __user__=user, __event_emitter__=emit))
    notes = [m["content"] for m in out["messages"] if m["role"] == "system"]
    return out, notes, events


def test_first_message_is_stored_and_the_model_is_told_there_is_no_record(owui):
    mod, f = owui
    out, notes, events = turn(f, "u-1", "Where did I move this year?")
    assert len(notes) == 1 and "don't have a record" in notes[0] and mod.MARKER in notes[0]
    assert any(e["type"] == "status" and "No memory answers" in e["data"]["description"] for e in events)


def test_later_turns_recall_what_the_user_said_with_sources(owui):
    mod, f = owui
    turn(f, "u-1", "I moved to Pune in March for my new job.")
    out, notes, events = turn(f, "u-1", "Which city did I move to for my new job?")
    assert len(notes) == 1 and "Pune" in notes[0] and notes[0].startswith(mod.MARKER)
    assert any(e["type"] == "status" and e["data"]["description"].startswith("Recalled") for e in events)
    assert any(e["type"] == "source" and "Pune" in e["data"]["document"][0] for e in events)


def test_users_never_see_each_others_memories(owui):
    _, f = owui
    turn(f, "u-1", "I moved to Pune in March for my new job.")
    _, notes, _ = turn(f, "u-2", "Which city did I move to for my new job?")
    assert "Pune" not in notes[0]


def test_one_note_per_turn_and_the_chat_survives_recallix_being_down(owui):
    mod, f = owui
    turn(f, "u-1", "I moved to Pune in March for my new job.")
    out, notes, _ = turn(f, "u-1", "Which city did I move to for my new job?", history=[{"role": "system", "content": f"{mod.MARKER}\nold note"}])
    assert len(notes) == 1 and "old note" not in notes[0]

    def down(*a, **k):
        raise ConnectionError("refused")
    mod.requests.post = down
    out, notes, events = turn(f, "u-1", "Which city did I move to for my new job?")
    assert notes == [] and out["messages"][-1]["content"] == "Which city did I move to for my new job?"
    assert any("unavailable" in e["data"]["description"] for e in events)


def test_a_user_can_switch_memory_off(owui):
    mod, f = owui
    off = f.UserValves(enabled=False)
    out, notes, events = turn(f, "u-3", "I moved to Pune in March for my new job.", user_valves=off)
    assert notes == [] and events == []
    _, notes, _ = turn(f, "u-3", "Which city did I move to for my new job?")
    assert "Pune" not in notes[0]  # nothing was stored while it was off


def test_user_settings_work_as_an_object_or_a_dict(owui):
    mod, f = owui
    for valves in (f.UserValves(enabled=False), {"enabled": False}):
        out, notes, events = turn(f, "u-4", "I moved to Pune in March for my new job.", user_valves=valves)
        assert notes == [] and events == []
    assert mod._uv({"mode": "cautious"}, "mode", "") == "cautious" and mod._uv(None, "mode", "x") == "x"
