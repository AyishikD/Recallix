"""The project was called Recallix before it became Recallic: old setting names, export files and the MCP command keep working."""
import json
import os
import subprocess
import sys

import requests

from tests.conftest import ROOT, EngineProcess, _free_port


def test_old_setting_names_still_configure_the_backend():
    code = ("import os; import backend; "
            "print(os.environ.get('RECALLIC_STORAGE_ROOT'), os.environ.get('RECALLIC_CONFIG'))")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RECALLIC_", "RECALLIX_"))}
    env.update(RECALLIX_STORAGE_ROOT="/tmp/old-name-root", RECALLIX_CONFIG="/tmp/old.yaml", RECALLIC_CONFIG="/tmp/new.yaml")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["/tmp/old-name-root", "/tmp/new.yaml"]  # the new name wins when both are set


def test_engine_accepts_the_old_setting_names(engine_process):
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RECALLIC_", "RECALLIX_"))}
    env.update(RECALLIX_INTERNAL_KEY="old-name-key-0123456789", RECALLIX_ENGINE_PORT=str(port),
               RECALLIX_ENGINE_HOST="127.0.0.1")
    proc = subprocess.Popen([engine_process.binary], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        eng = EngineProcess(engine_process.binary)
        eng.port, eng.url, eng.key, eng.proc = port, f"http://127.0.0.1:{port}", "old-name-key-0123456789", proc
        import time
        for _ in range(100):
            try:
                if requests.get(f"{eng.url}/health", timeout=0.5).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.05)
        # started on the old-name port, and the old-name key is the one it checks
        assert requests.get(f"{eng.url}/status", headers={"X-Internal-Key": "old-name-key-0123456789"}, timeout=5).status_code == 200
        assert requests.get(f"{eng.url}/status", headers={"X-Internal-Key": "wrong-key-0123456789"}, timeout=5).status_code in (401, 403)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_exports_made_before_the_rename_still_import(client, api_keys):
    old = {"format": "recallix-memory", "version": 1, "memories": [
        {"content": "I moved to Pune in March.", "timestamp": 1_700_000_000}]}
    for fmt in ("recallic", "recallix"):
        r = client.post("/memory/import", json={"format": fmt, "data": old, "user_id": f"old-{fmt}"},
                        headers=api_keys["admin"])
        assert r.status_code == 200, r.text
        assert r.json()["data"]["stored"] == 1
    exported = client.get("/memory/export", params={"user_id": "old-recallic"}, headers=api_keys["admin"]).json()["data"]
    assert exported["format"] == "recallic-memory"


def test_old_mcp_command_name_is_kept():
    text = (ROOT / "sdk" / "python" / "pyproject.toml").read_text()
    assert 'recallix-mcp = "recallic.mcp_server:main"' in text and 'recallic-mcp = "recallic.mcp_server:main"' in text
