"""POST /memory/store_batch and MemoryManager.store_many: one request, same result as storing one by one."""
import sqlite3
import time

from tests.test_security import _setting


def _rows(mm, user_id):
    conn = sqlite3.connect(mm.episodic.db_path)
    rows = conn.execute("SELECT content, timestamp, importance, metadata FROM episodic_events WHERE user_id = ? "
                        "ORDER BY id", (user_id,)).fetchall()
    conn.close()
    return rows


MESSAGES = [
    {"content": "I moved to Pune in March.", "timestamp": 1_700_000_300},
    {"content": "My sister Mira lives in Goa.", "timestamp": 1_700_000_100},
    {"content": "I started learning the violin.", "timestamp": 1_700_000_200},
    {"content": "My sister Mira lives in Goa.", "timestamp": 1_700_000_400},  # repeated: skipped
]


def test_store_many_matches_storing_one_by_one_in_time_order(engine):
    from backend.api.deps import get_memory_manager
    mm = get_memory_manager()
    for m in sorted(MESSAGES, key=lambda m: m["timestamp"]):
        mm.store("one", m["content"], timestamp=m["timestamp"])
    mm.semantic._cache.clear()  # the one-by-one run cached these texts
    calls = []
    real = mm.semantic.router.get_embeddings
    mm.semantic.router.get_embeddings = lambda texts: calls.append(len(texts)) or real(texts)
    out = mm.store_many("many", MESSAGES)
    assert calls == [3]  # one embedding call for the three distinct texts
    assert [r["status"] for r in out] == ["stored", "stored", "stored", "skipped_duplicate"]  # input order
    assert out[3]["memory_id"] == out[1]["memory_id"]
    one, many = _rows(mm, "one"), _rows(mm, "many")
    assert [r[0] for r in many] == ["My sister Mira lives in Goa.", "I started learning the violin.", "I moved to Pune in March."]
    assert [r[:3] for r in one] == [r[:3] for r in many]


def test_store_batch_endpoint(client, api_keys):
    h = api_keys["service"]
    body = {"user_id": "end-user-1", "agent_id": "adk",
            "items": [{**m, "metadata": {"speaker": "User"}} for m in MESSAGES]}
    r = client.post("/memory/store_batch", json=body, headers=h)
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["stored"] == 3 and data["skipped_duplicates"] == 1
    assert [x["status"] for x in data["results"]] == ["stored", "stored", "stored", "skipped_duplicate"]
    rec = client.post("/memory/recall", json={"query": "Where does Mira live?", "user_id": "end-user-1", "agent_id": "adk"},
                      headers=h).json()["data"]
    assert any("Goa" in m["content"] for m in rec["memories"])
    # a user key stores only for itself
    assert client.post("/memory/store_batch", json={**body, "user_id": "bob"}, headers=api_keys["alice"]).status_code == 403


def test_store_batch_limits(client, api_keys, monkeypatch):
    h = api_keys["alice"]
    _setting(monkeypatch, {"api.max_batch_items": 2})
    assert client.post("/memory/store_batch", json={"items": [{"content": f"note {i}"} for i in range(3)]},
                       headers=h).status_code == 413
    assert client.post("/memory/store_batch", json={"items": []}, headers=h).status_code == 422
    far = {"items": [{"content": "ok"}, {"content": "later", "timestamp": time.time() + 3 * 86400}]}
    r = client.post("/memory/store_batch", json=far, headers=h)
    assert r.status_code == 422 and "items[1]" in r.text
