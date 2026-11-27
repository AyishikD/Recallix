"""
Against a running Recallix server (skipped otherwise):
    RECALLIX_URL=http://127.0.0.1:8000 RECALLIX_SERVICE_KEY=rx_... pytest integrations/google-adk/tests
"""
import asyncio
import os
import time
import uuid

import pytest

URL, KEY = os.environ.get("RECALLIX_URL"), os.environ.get("RECALLIX_SERVICE_KEY")
pytestmark = pytest.mark.skipif(not (URL and KEY), reason="needs a running Recallix (RECALLIX_URL, RECALLIX_SERVICE_KEY)")


def run(coro):
    return asyncio.run(coro)


def session_with(app, user, turns):
    from google.adk.events.event import Event
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    svc = InMemorySessionService()
    s = run(svc.create_session(app_name=app, user_id=user))
    for author, text, ts in turns:
        run(svc.append_event(s, Event(author=author, timestamp=ts, invocation_id="i",
                                      content=types.Content(role="user" if author == "user" else "model",
                                                            parts=[types.Part(text=text)]))))
    return run(svc.get_session(app_name=app, user_id=user, session_id=s.id))


@pytest.fixture
def memory():
    from recallix_adk import RecallixMemoryService
    return RecallixMemoryService(api_key=KEY, base_url=URL)


def test_a_session_is_stored_with_its_dates_and_found_again(memory):
    app, user = "adk-test", f"u-{uuid.uuid4().hex[:8]}"
    month_ago = time.time() - 30 * 86400
    s = session_with(app, user, [("user", "I moved to Pune in March for my new job.", month_ago),
                                 ("assistant_agent", "Congratulations on the move!", month_ago + 5)])
    run(memory.add_session_to_memory(s))
    run(memory.add_session_to_memory(s))  # adding a session again stores nothing twice
    out = run(memory.search_memory(app_name=app, user_id=user, query="Where did I move?"))
    texts = [m.content.parts[0].text for m in out.memories]
    assert any("Pune" in t for t in texts)
    assert sum("Pune" in t for t in texts) == 1
    assert not any("Congratulations" in t for t in texts)  # only what the user said is stored by default
    pune = next(m for m in out.memories if "Pune" in m.content.parts[0].text)
    assert pune.timestamp.startswith(time.strftime("%Y-%m-%d", time.gmtime(month_ago)))


def test_no_record_means_no_memories_and_users_and_apps_are_separate(memory):
    app, user = "adk-test", f"u-{uuid.uuid4().hex[:8]}"
    run(memory.add_events_to_memory(app_name=app, user_id=user,
                                    events=session_with(app, user, [("user", "I moved to Pune in March for my new job.", time.time())]).events))
    assert run(memory.search_memory(app_name=app, user_id=f"other-{user}", query="Where did I move?")).memories == []
    assert run(memory.search_memory(app_name="other-app", user_id=user, query="Where did I move?")).memories == []
    assert run(memory.search_memory(app_name=app, user_id=user, query="What is my sister's name?")).memories == []


def test_delete_a_memory(memory):
    app, user = "adk-test", f"u-{uuid.uuid4().hex[:8]}"
    run(memory.add_events_to_memory(app_name=app, user_id=user,
                                    events=session_with(app, user, [("user", "My locker code is 4417.", time.time())]).events))
    found = run(memory.search_memory(app_name=app, user_id=user, query="What is my locker code?")).memories
    assert found
    run(memory.delete_memory(app_name=app, user_id=user, memory_id=found[0].id))
    assert run(memory.search_memory(app_name=app, user_id=user, query="What is my locker code?")).memories == []
