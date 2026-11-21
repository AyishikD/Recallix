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
