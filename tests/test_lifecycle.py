import os
import sqlite3
import threading
import time

from tests.conftest import wait_for


def _store(client, headers, content, **extra):
    r = client.post("/memory/store", json={"content": content, **extra}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def _recall(client, headers, query, **extra):
    r = client.post("/memory/recall", json={"query": query, **extra}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def test_fresh_storage_root_is_created(tmp_path):
    from backend.api.deps import get_memory_manager
    get_memory_manager()
    root = tmp_path / "storage"
    for sub in ("sqlite_db", "timeline_store", "world_state_store", "archive_store"):
        assert (root / sub).is_dir(), sub


def test_write_is_searchable_immediately(client, api_keys, engine):
    _store(client, api_keys["alice"], "Remember this: the Hydra cluster has 64 GPUs.")
    out = _recall(client, api_keys["alice"], "How many GPUs does Hydra have?")
    assert "64 GPUs" in out["memories"][0]["content"]


def test_delete_removes_every_copy(client, api_keys, engine):
    h = api_keys["alice"]
    mid = _store(client, h, "Important: my passport number is X1234567, expires 2031.")["memory_id"]
    assert engine.get("/namespaces")["alice"]["live"] == 1

    assert client.delete(f"/memory/{mid}", headers=h).status_code == 200
    assert engine.get("/namespaces")["alice"]["live"] == 0
    assert not any("X1234567" in m["content"] for m in _recall(client, h, "passport number")["memories"])
    assert not any("X1234567" in e["content"] for e in client.get("/memory/timeline", headers=h).json()["data"])
    from backend.api.deps import get_memory_manager
    assert not any("X1234567" in f["fact"] for f in get_memory_manager().long_term.get("alice"))

    assert client.delete(f"/memory/{mid}", headers=h).status_code == 404
    other = _store(client, api_keys["bob"], "Bob's note about tea")["memory_id"]
    assert client.delete(f"/memory/{other}", headers=h).status_code == 404  # not alice's


def test_clearing_one_user_keeps_others(client, api_keys, engine):
    _store(client, api_keys["alice"], "Alice likes tennis.")
    _store(client, api_keys["bob"], "Bob likes chess.")
    r = client.post("/internal/clear", json={"user_id": "alice"}, headers=api_keys["admin"])
    assert r.status_code == 200 and r.json()["data"]["memories_deleted"] == 1
    spaces = engine.get("/namespaces")
    assert "alice" not in spaces and spaces["bob"]["live"] == 1
    assert "chess" in _recall(client, api_keys["bob"], "What does Bob like?")["memories"][0]["content"]


def test_dedupe_persists_across_restarts(client, api_keys):
    from backend.api import deps
    first = _store(client, api_keys["alice"], "My name is Priya.")
    deps._memory_manager = None  # simulate a process restart (new manager, same storage)
    second = _store(client, api_keys["alice"], "My name is Priya.")
    assert first["status"] == "stored" and second["status"] == "skipped_duplicate"
    assert second["memory_id"] == first["memory_id"]
    # same content from a different agent's private scope is a different memory
    third = _store(client, api_keys["alice"], "My name is Priya.", agent_id="other")
    assert third["status"] == "stored"


def test_index_rebuilt_after_engine_restart(client, api_keys, engine):
    h = api_keys["alice"]
    for text in ["The staging server runs Ubuntu 22.04.", "Lunch was good.", "Standup is at 10am."]:
        _store(client, h, text)
    engine.restart()  # in-memory indexes are gone
    assert engine.get("/namespaces") == {}
    _recall(client, h, "What OS does staging run?")  # first contact with the new instance triggers a rebuild
    assert wait_for(lambda: engine.get("/namespaces").get("alice", {}).get("live") == 3)
    # the rebuilt index answers vector queries again
    from backend.api.deps import get_memory_manager
    mm = get_memory_manager()
    hits = mm.semantic.search("alice", "The staging server runs Ubuntu 22.04.", top_k=1)
    assert hits and "Ubuntu" in mm.episodic.get_by_id("alice", hits[0][0])["content"] and hits[0][1] > 0.99


def test_startup_sync_rebuilds_missing_indexes(engine, api_keys):
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        for text in ["Kenji lives in Osaka.", "The retro is on Thursday."]:
            _store(c, api_keys["alice"], text)
    engine.post("/drop_all")
    from backend.api import deps
    deps.reset()
    with TestClient(server.app):  # lifespan runs sync()
        assert engine.get("/namespaces")["alice"]["live"] == 2


def test_embedding_model_change_never_mixes_dimensions(client, api_keys, engine, monkeypatch):
    import backend.brain.models.model_router as mr
    from backend.api import deps
    from backend.brain.memory import index_sync
    from tests.conftest import hash_embedding
    h = api_keys["alice"]
    for text in ["Anita lives in Bangalore.", "Marco works at Stripe.", "The offsite is on 12 May."]:
        _store(client, h, text)
    assert engine.get("/namespaces")["alice"]["dim"] == 64

    # switch to a different embedding model (different name and dimension)
    monkeypatch.setattr(mr.ModelRouter, "get_embedding", lambda self, t: hash_embedding(t, dim=32))
    real_init = mr.ModelRouter.__init__

    def init(self):
        real_init(self)
        self.embedding_model = "other-model"
    monkeypatch.setattr(mr.ModelRouter, "__init__", init)
    deps._memory_manager = None
    mm = deps.get_memory_manager()
    assert mm.episodic.stale_embedding_count(mm.semantic.model_name) == 3

    # recall works (no dimension errors) before any backfill: lexical + no stale vectors
    assert "Bangalore" in _recall(client, h, "Where does Anita live?")["memories"][0]["content"]

    index_sync.backfill(mm.episodic, mm.semantic)
    index_sync.sync(mm.episodic, mm.semantic)
    spaces = engine.get("/namespaces")
    assert spaces["alice"]["dim"] == 32 and spaces["alice"]["live"] == 3
    assert mm.episodic.stale_embedding_count(mm.semantic.model_name) == 0


def test_background_jobs_produce_state_insights_and_forget(client, api_keys, engine):
    h = api_keys["admin"]
    a = api_keys["alice"]
    _store(client, a, "I live in Pune.")
    _store(client, a, "Important: my blood group is O negative, remember this.")
    _store(client, a, "Need to renew my passport before 1 June.")
    low = _store(client, a, "ok")

    assert client.get("/memory/timeline", headers=a).json()["data"]
    client.post("/internal/jobs/run", json={"job": "state_inference", "user_id": "alice"}, headers=h)
    state = client.get("/memory/world-state", headers=a).json()["data"]
    assert state["location"]["value"] == "Pune"

    client.post("/internal/jobs/run", json={"job": "replay", "user_id": "alice"}, headers=h)
    insights = client.get("/memory/insights", headers=a).json()["data"]
    assert insights and "blood group" in insights[0]["summary"]

    pred = client.get("/memory/prediction", headers=a).json()["data"]
    assert pred["based_on"] == 4 and len(pred["next_topics"]) == 3

    # age the low-importance memory past the delete threshold, then run forgetting
    from backend.api.deps import get_memory_manager
    db = get_memory_manager().episodic.db_path
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE episodic_events SET timestamp = ? WHERE id = ?", (time.time() - 40 * 86400, low["memory_id"]))
    before = engine.get("/namespaces")["alice"]["live"]
    result = client.post("/internal/jobs/run", json={"job": "forgetting"}, headers=h).json()["data"]
    assert result["deleted"] == 1
    assert engine.get("/namespaces")["alice"]["live"] == before - 1


def test_recall_feedback_is_logged_and_fittable(client, api_keys):
    from backend.brain.memory_ranking import fit_ranker
    from backend.brain.memory_ranking.ranker import RecallLog
    from collections import defaultdict
    h = api_keys["alice"]
    _store(client, h, "Project Heron deadline is April 3.")
    _store(client, h, "The coffee machine is broken.")
    for _ in range(3):
        out = _recall(client, h, "When is Project Heron due?")
        right = [m["id"] for m in out["memories"] if "Heron" in m["content"]]
        r = client.post("/memory/feedback", json={"recall_id": out["recall_id"], "relevant_memory_ids": right}, headers=h)
        assert r.status_code == 200
    assert client.post("/memory/feedback", json={"recall_id": "nope", "relevant_memory_ids": []},
                       headers=h).status_code == 404
    groups = defaultdict(list)
    for rid, f, y in RecallLog().labelled():
        groups[rid].append((f, y))
    model = fit_ranker.fit_model(list(groups.values()))
    assert 0.0 < model["threshold"] < 1.0 and set(model["weights"]) >= {"bias", "semantic"}


def test_concurrent_stores_and_recalls(client, api_keys, engine):
    h = api_keys["alice"]
    errors = []

    def worker(i):
        try:
            for j in range(5):
                r = client.post("/memory/store", json={"content": f"Worker {i} fact {j}: item code W{i}-{j}"}, headers=h)
                assert r.status_code in (200, 429), r.text
                r = client.post("/memory/recall", json={"query": f"item code W{i}-{j}"}, headers=h)
                assert r.status_code in (200, 429), r.text  # api.reads_per_second_per_user
            # identical content raced from many threads must be stored once
            client.post("/memory/store", json={"content": "Shared duplicate note"}, headers=h)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    from backend.api.deps import get_memory_manager
    db = get_memory_manager().episodic.db_path
    with sqlite3.connect(db) as conn:
        dup = conn.execute("SELECT COUNT(*) FROM episodic_events WHERE content = 'Shared duplicate note'").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM episodic_events WHERE user_id = 'alice'").fetchone()[0]
    assert dup == 1
    assert wait_for(lambda: engine.get("/namespaces")["alice"]["live"] == total)


def test_no_llm_on_any_path(client, api_keys):
    # the autouse fixture makes ModelRouter.route raise; exercise every endpoint
    a, h = api_keys["alice"], api_keys["admin"]
    _store(client, a, "I prefer tea over coffee.")
    _recall(client, a, "What do I drink?")
    for path in ("/memory/timeline", "/memory/insights", "/memory/world-state", "/memory/prediction",
                 "/memory/plan", "/memory/stats", "/memory/profile", "/memory/system/stats"):
        assert client.get(path, headers=a).status_code == 200, path
    for job in ("state_inference", "prediction", "replay", "forgetting"):
        assert client.post("/internal/jobs/run", json={"job": job}, headers=h).status_code == 200, job
    assert os.environ.get("RECALLIX_INTERNAL_KEY")
