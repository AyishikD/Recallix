import json
import subprocess

import requests

from tests.conftest import ROOT


def test_missing_and_invalid_keys(client):
    assert client.post("/memory/recall", json={"query": "x"}).status_code == 401
    assert client.post("/memory/recall", json={"query": "x"}, headers={"X-API-Key": "rx_nope"}).status_code == 403


def test_user_key_cannot_act_as_another_user(client, api_keys):
    r = client.post("/memory/store", json={"content": "Alice secret", "user_id": "bob"}, headers=api_keys["alice"])
    assert r.status_code == 403
    r = client.post("/memory/recall", json={"query": "secret", "user_id": "bob"}, headers=api_keys["alice"])
    assert r.status_code == 403
    assert client.delete("/memory/1?user_id=bob", headers=api_keys["alice"]).status_code == 403


def test_service_key_may_act_for_users(client, api_keys):
    r = client.post("/memory/store", json={"content": "The Orion launch date is March 4.", "user_id": "end-user-7"},
                    headers=api_keys["service"])
    assert r.status_code == 200 and r.json()["data"]["status"] == "stored"
    r = client.post("/memory/recall", json={"query": "When is the Orion launch?", "user_id": "end-user-7"},
                    headers=api_keys["service"])
    assert "March 4" in r.json()["data"]["memories"][0]["content"]


def test_internal_endpoints_require_admin(client, api_keys):
    for who in ("alice", "service"):
        assert client.post("/internal/clear", json={"user_id": "all"}, headers=api_keys[who]).status_code == 403
        assert client.post("/internal/reindex", json={}, headers=api_keys[who]).status_code == 403
    assert client.post("/internal/clear", json={"user_id": "all"}).status_code == 401
    assert client.post("/internal/clear", json={"user_id": "nobody"}, headers=api_keys["admin"]).status_code == 200


def test_keys_are_stored_hashed(api_keys):
    from backend.api.keys import keys_path
    text = open(keys_path()).read()
    for headers in api_keys.values():
        assert headers["X-API-Key"] not in text


def test_legacy_plaintext_key_file_is_migrated():
    from backend.api import keys
    with open(keys.keys_path(), "w") as f:
        json.dump({"keys": {"legacy-plain-key": {"user_id": "old-user"}}}, f)
    keys._cache.update(mtime=None, keys={})
    principal = keys.lookup("legacy-plain-key")
    assert principal["user_id"] == "old-user" and principal["role"] == "user"
    assert "legacy-plain-key" not in open(keys.keys_path()).read()


def test_engine_requires_internal_key(engine):
    assert requests.post(f"{engine.url}/search", json={"namespace": "x", "query": [1.0]}).status_code == 401
    assert requests.post(f"{engine.url}/search", json={"namespace": "x", "query": [1.0]},
                         headers={"X-Internal-Key": "wrong-key-wrong-key"}).status_code == 401
    assert requests.get(f"{engine.url}/health").status_code == 200


def test_engine_refuses_to_start_without_key(engine_process):
    import os
    env = {k: v for k, v in os.environ.items() if k != "RECALLIX_INTERNAL_KEY"}
    proc = subprocess.run([engine_process.binary], env=env, capture_output=True, timeout=10)
    assert proc.returncode != 0
    assert b"RECALLIX_INTERNAL_KEY" in proc.stderr


def test_no_hardcoded_secrets_in_repository():
    # Former hardcoded values, assembled so this file does not contain them literally.
    banned = ["Recallix-Core-" + "8892", "local_" + "dev_key"]
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    offenders = []
    for path in tracked:
        if path.endswith((".png", ".webp", ".jpg", ".ico", ".svg", ".woff", ".woff2", "package-lock.json")):
            continue
        try:
            text = (ROOT / path).read_text(errors="ignore")
        except (FileNotFoundError, IsADirectoryError):
            continue
        offenders += [f"{path}: {b}" for b in banned if b in text]
    assert offenders == []


def test_rate_limit(client, api_keys, monkeypatch):
    from backend import settings
    real_get = settings.get
    monkeypatch.setattr(settings, "get", lambda p, d=None: 3 if p == "api.writes_per_second_per_user" else real_get(p, d))
    codes = [client.post("/memory/store", json={"content": f"note number {i}"}, headers=api_keys["alice"]).status_code
             for i in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]


def _setting(monkeypatch, overrides):
    from backend import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda path, default=None: overrides[path] if path in overrides else real(path, default))


def test_oversized_requests_are_refused(client, api_keys, monkeypatch):
    h = api_keys["alice"]
    _setting(monkeypatch, {"api.max_request_bytes": 2000, "api.max_metadata_bytes": 100, "api.max_import_items": 2})
    r = client.post("/memory/store", content=b'{"content": "' + b"x" * 5000 + b'"}',
                    headers={**h, "Content-Type": "application/json"})
    assert r.status_code == 413
    r = client.post("/memory/store", json={"content": "hello", "metadata": {"blob": "y" * 500}}, headers=h)
    assert r.status_code == 422 and "metadata" in r.text
    r = client.post("/memory/import", json={"format": "text", "data": "one\ntwo\nthree"}, headers=h)
    assert r.status_code == 413


def test_recalls_are_rate_limited_and_deep_recall_costs_more(client, api_keys, monkeypatch):
    h = api_keys["alice"]
    _setting(monkeypatch, {"api.reads_per_second_per_user": 3})
    assert client.post("/memory/recall", json={"query": "x", "deep": True}, headers=h).status_code == 429  # costs 4 > 3
    codes = [client.post("/memory/recall", json={"query": "x"}, headers=h).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_user_and_agent_ids_are_bounded(client, api_keys):
    s = api_keys["service"]
    for bad in ("a" * 129, "eve\nadmin", "x\x00y"):
        assert client.post("/memory/store", json={"content": "hi", "user_id": bad}, headers=s).status_code == 422
        assert client.get("/memory/stats", params={"user_id": bad}, headers=s).status_code == 422
    assert client.post("/memory/store", json={"content": "hi", "agent_id": "a" * 129}, headers=api_keys["alice"]).status_code == 422
    from backend.api import keys
    import pytest
    with pytest.raises(ValueError):
        keys.create("bad\nuser")


def test_agent_list_and_system_stats_show_only_your_own(client, api_keys):
    a, b = api_keys["alice"], api_keys["bob"]
    client.post("/agents/store", json={"content": "Alice planner note", "agent_id": "alice-planner"}, headers=a)
    client.post("/agents/store", json={"content": "Bob coder note", "agent_id": "bob-coder"}, headers=b)
    mine = client.get("/agents/list", headers=a).json()["data"]
    assert [x["agent_id"] for x in mine] == ["alice-planner"] and mine[0]["memories"] == 1
    engine = client.get("/memory/system/stats", headers=a).json()["data"]["engine"]
    assert "namespaces" not in engine and engine.get("status")              # no engine-wide counts for users
    assert "namespaces" in client.get("/memory/system/stats", headers=api_keys["admin"]).json()["data"]["engine"]


def test_one_import_at_a_time_per_user():
    from fastapi import HTTPException
    import pytest
    from backend.api.middleware.auth import one_import_at_a_time
    with one_import_at_a_time("u1"):
        with pytest.raises(HTTPException) as e:
            with one_import_at_a_time("u1"):
                pass
        assert e.value.status_code == 429
        with one_import_at_a_time("u2"):   # other users are not blocked
            pass
    with one_import_at_a_time("u1"):       # released afterwards
        pass


def test_health_needs_no_key_and_reports_the_engine(client, monkeypatch):
    out = client.get("/health")
    assert out.status_code == 200 and out.json() == {"status": "ok", "api": "ok", "engine": "ok"}
    import backend.utils.internal_client as ic

    def down(path):
        raise ic.EngineError("GET /health -> connection refused")
    monkeypatch.setattr(ic, "engine_get", down)
    out = client.get("/health")
    assert out.status_code == 503 and out.json()["engine"] == "unreachable"


def test_store_keeps_the_original_time_and_refuses_the_far_future(client, api_keys):
    import time
    h = api_keys["alice"]
    then = time.time() - 30 * 86400
    out = client.post("/memory/store", json={"content": "Signed the lease for the flat in Pune.", "timestamp": then}, headers=h)
    assert out.status_code == 200
    mid = out.json()["data"]["memory_id"]
    rec = client.post("/memory/recall", json={"query": "Signed the lease for the flat in Pune?", "keep_on_decline": True}, headers=h).json()["data"]
    m = next(m for m in rec["memories"] if m["id"] == mid)
    assert abs(m["timestamp"] - then) < 1
    assert client.post("/memory/store", json={"content": "x", "timestamp": time.time() + 3 * 86400}, headers=h).status_code == 422
