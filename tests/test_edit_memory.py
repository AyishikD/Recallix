"""PUT /memory/{id} and MemoryManager.edit: replace a memory's text, keep its time; the old version is deleted."""
import sqlite3

THEN = 1_700_000_000.0


def _store(client, h, content, **extra):
    r = client.post("/memory/store", json={"content": content, "timestamp": THEN, **extra}, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["data"]["memory_id"]


def _all_rows(mm, user_id):
    with sqlite3.connect(mm.episodic.db_path) as c:
        return c.execute("SELECT id, content, timestamp, metadata, status FROM episodic_events WHERE user_id = ?",
                         (user_id,)).fetchall()


def test_edit_replaces_the_text_and_keeps_the_time(client, api_keys):
    from backend.api.deps import get_memory_manager
    h = api_keys["alice"]
    mid = _store(client, h, "I moved to Pune in March.", metadata={"speaker": "User"})
    r = client.put(f"/memory/{mid}", json={"content": "I moved to Mumbai in March."}, headers=h)
    assert r.status_code == 200, r.text
    out = r.json()["data"]
    assert out["status"] == "edited" and out["previous_memory_id"] == mid and out["memory_id"] != mid
    assert out["conflicts"] == []  # not a conflict with its own old version
    assert out["receipt"]["reason"] == "edited" and out["receipt"]["all_copies_gone"]

    rows = _all_rows(get_memory_manager(), "alice")
    assert [r[1] for r in rows] == ["I moved to Mumbai in March."]  # nothing of the old text is left
    assert rows[0][0] == out["memory_id"] and abs(rows[0][2] - THEN) < 1e-6
    assert '"speaker": "User"' in rows[0][3]  # client metadata carried over

    rec = client.post("/memory/recall", json={"query": "Where did I move to in March?", "keep_on_decline": True},
                      headers=h).json()["data"]
    contents = [m["content"] for m in rec["memories"]]
    assert "I moved to Mumbai in March." in contents and not any("Pune" in c for c in contents)

    deletions = client.get("/memory/deletions", headers=h).json()["data"]
    assert any(d.get("memory_id") == mid and d.get("reason") == "edited" for d in
               (deletions if isinstance(deletions, list) else deletions.get("receipts", [])))
    # the old id is gone
    assert client.put(f"/memory/{mid}", json={"content": "x"}, headers=h).status_code == 404


def test_edit_is_scoped_to_the_owner(client, api_keys):
    mid = _store(client, api_keys["alice"], "My locker code is in the blue notebook.")
    assert client.put(f"/memory/{mid}", json={"content": "hacked"}, headers=api_keys["bob"]).status_code == 404
    assert client.put(f"/memory/{mid}", json={"content": "hacked", "user_id": "alice"},
                      headers=api_keys["bob"]).status_code == 403


def test_unchanged_and_merged_edits(client, api_keys):
    from backend.api.deps import get_memory_manager
    h = api_keys["alice"]
    a = _store(client, h, "I started learning the violin.")
    out = client.put(f"/memory/{a}", json={"content": "I started learning the violin."}, headers=h).json()["data"]
    assert out["status"] == "unchanged" and out["memory_id"] == a
    b = _store(client, h, "My sister Mira lives in Goa.")
    out = client.put(f"/memory/{a}", json={"content": "My sister Mira lives in Goa."}, headers=h).json()["data"]
    assert out["status"] == "merged" and out["memory_id"] == b
    assert [r[0] for r in _all_rows(get_memory_manager(), "alice")] == [b]


def test_edit_can_set_a_new_time_and_metadata(client, api_keys):
    from backend.api.deps import get_memory_manager
    h = api_keys["alice"]
    mid = _store(client, h, "Booked the flight to Lisbon.")
    out = client.put(f"/memory/{mid}", json={"content": "Booked the flight to Porto.", "timestamp": THEN + 86400,
                                              "metadata": {"source": "fix"}}, headers=h).json()["data"]
    row = [r for r in _all_rows(get_memory_manager(), "alice") if r[0] == out["memory_id"]][0]
    assert abs(row[2] - (THEN + 86400)) < 1e-6 and '"source": "fix"' in row[3]


def test_failed_store_puts_the_old_text_back(engine, monkeypatch):
    from backend.api.deps import get_memory_manager
    mm = get_memory_manager()
    mid = mm.store("carol", "The wifi password is on the fridge.", timestamp=THEN)["memory_id"]
    real = mm.store

    def broken(user_id, message, **kw):
        if message == "new text":
            raise RuntimeError("disk full")
        return real(user_id, message, **kw)
    monkeypatch.setattr(mm, "store", broken)
    try:
        mm.edit("carol", mid, "new text")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the failure should be reported")
    rows = _all_rows(mm, "carol")
    assert [r[1] for r in rows] == ["The wifi password is on the fridge."] and abs(rows[0][2] - THEN) < 1e-6
