"""
Test fixtures.

* The real C++ engine binary is started once per session on a free port with a
  random RECALLIX_INTERNAL_KEY (set RECALLIX_ENGINE_BIN, default
  backend/infra_cpp/build/memory_service). Tests needing it are skipped if it
  has not been built.
* Every test gets a fresh storage root and fresh singletons.
* Embeddings come from a deterministic hashed bag-of-words embedder; any LLM
  call fails the test.
"""
import hashlib
import os
import pathlib
import secrets
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RECALLIX_INTERNAL_KEY", secrets.token_hex(16))


def hash_embedding(text, dim=64):
    from backend.brain.memory_ranking.importance_ranker import content_terms
    v = np.zeros(dim)
    for t in content_terms(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1
    return v.tolist() if v.any() else []


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EngineProcess:
    def __init__(self, binary):
        self.binary = binary
        self.key = secrets.token_hex(16)
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = None

    def start(self):
        env = dict(os.environ, RECALLIX_INTERNAL_KEY=self.key, RECALLIX_ENGINE_PORT=str(self.port),
                   RECALLIX_ENGINE_HOST="127.0.0.1")
        self.proc = subprocess.Popen([self.binary], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if requests.get(f"{self.url}/health", timeout=0.5).status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.05)
        raise RuntimeError("engine did not start")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    def restart(self):
        self.stop()
        self.start()

    def get(self, path):
        return requests.get(f"{self.url}{path}", headers={"X-Internal-Key": self.key}, timeout=5).json()

    def post(self, path, body=None):
        return requests.post(f"{self.url}{path}", json=body or {}, headers={"X-Internal-Key": self.key}, timeout=5).json()


@pytest.fixture(scope="session")
def engine_process():
    binary = os.environ.get("RECALLIX_ENGINE_BIN", str(ROOT / "backend/infra_cpp/build/memory_service"))
    if not os.path.exists(binary):
        pytest.skip(f"engine binary not built ({binary})")
    eng = EngineProcess(binary)
    eng.start()
    yield eng
    eng.stop()


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    from backend import settings
    from backend.api import deps, keys
    import backend.brain.models.model_router as mr

    from backend.api.middleware import auth
    monkeypatch.setenv("RECALLIX_STORAGE_ROOT", str(tmp_path / "storage"))
    # the shipped ranking fit belongs to the default embedding model; the hash embedder has its own
    from backend.tools.sandbox import install_hash_fits
    (tmp_path / "storage").mkdir(parents=True, exist_ok=True)
    install_hash_fits(str(tmp_path / "storage"))
    settings.reload()
    deps.reset()
    keys._cache.update(mtime=None, keys={})
    monkeypatch.setattr(auth, "_limiter", auth._RateLimiter())
    # The answer checker downloads a model; tests use it only through explicit fakes.
    from backend.brain.features import answer_check
    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: None))
    monkeypatch.setattr(mr.ModelRouter, "get_embedding", lambda self, text: hash_embedding(text))

    def no_llm(self, *a, **k):
        raise AssertionError("an LLM was called")
    monkeypatch.setattr(mr.ModelRouter, "route", no_llm, raising=False)
    yield
    deps.reset()


@pytest.fixture
def engine(engine_process, monkeypatch):
    from backend import settings
    monkeypatch.setenv("RECALLIX_INTERNAL_KEY", engine_process.key)
    monkeypatch.setenv("RECALLIX_ENGINE_URL", engine_process.url)
    settings.reload()
    engine_process.post("/drop_all")
    return engine_process


@pytest.fixture
def client(engine):
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def api_keys():
    from backend.api import keys
    return {
        "admin": {"X-API-Key": keys.create("admin", "admin")},
        "alice": {"X-API-Key": keys.create("alice", "user")},
        "bob": {"X-API-Key": keys.create("bob", "user")},
        "service": {"X-API-Key": keys.create("svc", "service")},
    }


def wait_for(predicate, timeout=15.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
