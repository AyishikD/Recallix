import socket
import sys
import threading
import time

import pytest

from tests.conftest import ROOT


@pytest.fixture
def live_server(engine):
    import uvicorn
    import server
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(timeout=10)


def test_python_sdk_roundtrip(live_server, api_keys):
    sys.path.insert(0, str(ROOT / "sdk" / "python"))
    from recallix import Memory, RecallixError
    mem = Memory(api_key=api_keys["alice"]["X-API-Key"], base_url=live_server)

    stored = mem.store("The Orion launch date is March 4.", agent_id="planner", memory_type="shared")
    assert stored["status"] == "stored"
    out = mem.recall("When is the Orion launch?", agent_id="researcher")
    assert "March 4" in out["memories"][0]["content"]
    assert mem.feedback(out["recall_id"], [out["memories"][0]["id"]])["labelled_candidates"] >= 1
    assert mem.timeline()[0]["content"].startswith("The Orion")
    assert mem.delete(stored["memory_id"])["deleted"] == stored["memory_id"]

    with pytest.raises(RecallixError) as e:
        mem.recall("x", user_id="bob")  # user key may not act for another user
    assert e.value.status == 403


def test_rag_app_passwords_are_salted_and_legacy_rows_upgrade(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "rag_chat_app"))
    import hashlib
    import sqlite3
    import database
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "app.db"))
    database.init_db()
    assert database.register_user("ana", "pw1")
    h1 = sqlite3.connect(database.DB_PATH).execute("SELECT password_hash FROM users").fetchone()[0]
    assert h1.startswith("pbkdf2$") and "pw1" not in h1
    assert database.register_user("ben", "pw1")
    h2 = sqlite3.connect(database.DB_PATH).execute("SELECT password_hash FROM users WHERE username='ben'").fetchone()[0]
    assert h1 != h2  # salted
    assert database.login_user("ana", "pw1")[1] == "ana"
    assert database.login_user("ana", "wrong") is None

    conn = sqlite3.connect(database.DB_PATH)
    conn.execute("INSERT INTO users (username, password_hash) VALUES ('old', ?)", (hashlib.sha256(b"pw").hexdigest(),))
    conn.commit()
    assert database.login_user("old", "pw") is not None
    upgraded = conn.execute("SELECT password_hash FROM users WHERE username='old'").fetchone()[0]
    assert upgraded.startswith("pbkdf2$")
