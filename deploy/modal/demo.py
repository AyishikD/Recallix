"""
The public Recallic demo on Modal (https://modal.com): the real engine and memory pipeline in one container,
behind a small demo API that www.recallic.com/demo calls.

    uvx --from modal modal deploy deploy/modal/demo.py      # from the repo root; prints the demo's URL
    uvx --from modal modal serve deploy/modal/demo.py       # temporary, live-reloading copy for testing

Safe to leave public:
* No API keys exist in the container. The demo API is the only way in, and each browser session gets its own
  user id, so visitors never see each other's memories.
* Limits per session (messages, length) and per visitor (requests per minute).
* Memories live in the container's /tmp only. They are gone when it shuts down, a few minutes after the last
  visitor; nothing is written to disk that outlives the container.
* One container at most (max_containers=1), so cost has a ceiling. The models are baked into the image, so a
  cold start only loads them from local disk.
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGES = ["fastapi>=0.110", "uvicorn>=0.27", "pydantic>=2.5", "requests>=2.33.0", "numpy>=1.26", "pyyaml>=6.0",
            "python-dotenv>=1.2.2", "starlette>=1.3.1", "h11>=0.16.0", "urllib3>=2.8.0",
            "sentence-transformers>=5.6.0", "transformers>=5.10.0", "jinja2>=3.1.6"]

WARM_MODELS = """
import os
os.environ.setdefault("RECALLIC_STORAGE_ROOT", "/tmp/warm"); os.environ.setdefault("RECALLIC_INTERNAL_KEY", "build-only-not-a-secret")
from backend.brain.models.model_router import ModelRouter
from backend.brain.features.answer_check import AnswerChecker
assert ModelRouter().get_embedding("warm up"), "embedding model failed to load"
AnswerChecker.from_config()
print("models baked into the image")
"""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("cmake", "g++", "make")
    .pip_install("torch>=2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(*PACKAGES)
    .add_local_dir(ROOT / "backend" / "infra_cpp", "/src/engine", copy=True, ignore=["build", "build/**"])
    .run_commands(
        "cmake -S /src/engine -B /src/build -DCMAKE_BUILD_TYPE=Release -DRECALLIC_DISABLE_AVX2=ON -DRECALLIC_TSAN=OFF"
        " && cmake --build /src/build --target memory_service -j4 && cp /src/build/memory_service /usr/local/bin/")
    .env({"HF_HOME": "/models", "RECALLIC_EMBEDDING_PROVIDER": "local", "PYTHONPATH": "/app"})
    .add_local_dir(ROOT / "backend", "/app/backend", copy=True,
                   ignore=["infra_cpp/build", "infra_cpp/build/**", "storage", "storage/**", "**/__pycache__/**"])
    .add_local_file(ROOT / "config.yaml", "/app/config.yaml", copy=True)
    .run_commands(f"cd /app && python -c '{WARM_MODELS.strip()}'".replace("\n", "; ").replace("; ;", ";"))
)

app = modal.App("recallic-demo", image=image)

ALLOWED_ORIGINS = ["https://www.recallic.com", "https://recallic.com", "https://recallix.vercel.app",
                   "http://localhost:3000"]
MAX_MESSAGES = 30          # per session
MAX_CHARS = 400            # per message or question
PER_MINUTE = 30            # requests per visitor (IP) per minute


@app.cls(cpu=2.0, memory=2048, scaledown_window=300, timeout=60, max_containers=1)
@modal.concurrent(max_inputs=20)
class Demo:
    @modal.enter()
    def start(self):
        import os
        import secrets
        import subprocess
        import sys
        import time
        import urllib.request

        os.environ.update(RECALLIC_INTERNAL_KEY=secrets.token_hex(16), RECALLIC_ENGINE_HOST="127.0.0.1",
                          RECALLIC_ENGINE_PORT="8080", RECALLIC_ENGINE_URL="http://127.0.0.1:8080",
                          RECALLIC_STORAGE_ROOT="/tmp/recallic")
        os.makedirs("/tmp/recallic", exist_ok=True)
        self.engine = subprocess.Popen(["memory_service"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        sys.path.insert(0, "/app")
        os.chdir("/app")
        from backend.api.deps import get_memory_manager
        self.mm = get_memory_manager()
        self.mm.store("demo-warmup", "Warm-up message.", agent_id="demo")
        self.mm.retrieve("demo-warmup", "warm up?", agent_id="demo")  # loads every model before the first visitor

    @modal.exit()
    def stop(self):
        self.engine.terminate()

    @modal.asgi_app()
    def web(self):
        import collections
        import re
        import threading
        import time

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field

        api = FastAPI(title="Recallic demo", docs_url=None, redoc_url=None, openapi_url=None)
        api.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"])
        hits = collections.defaultdict(collections.deque)
        counts = collections.Counter()
        lock = threading.Lock()
        mm = self.mm

        def limit(request: Request):
            ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
            now = time.time()
            with lock:
                q = hits[ip]
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= PER_MINUTE:
                    raise HTTPException(429, "Too many requests from you in a minute; try again shortly.")
                q.append(now)

        def user(session: str) -> str:
            if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", session):
                raise HTTPException(422, "bad session id")
            return f"demo-{session}"

        class Store(BaseModel):
            session: str = Field(max_length=64)
            content: str = Field(min_length=1, max_length=MAX_CHARS)

        class Recall(BaseModel):
            session: str = Field(max_length=64)
            query: str = Field(min_length=1, max_length=MAX_CHARS)

        @api.get("/health")
        def health():
            return {"status": "ok"}

        @api.post("/store")
        def store(body: Store, request: Request):
            limit(request)
            uid = user(body.session)
            with lock:
                if counts[uid] >= MAX_MESSAGES:
                    raise HTTPException(429, f"This demo keeps at most {MAX_MESSAGES} messages per visitor.")
                counts[uid] += 1
            out = mm.store(uid, body.content, agent_id="demo")
            return {"status": out["status"], "memory_id": out["memory_id"], "ms": out.get("total_ms")}

        @api.post("/recall")
        def recall(body: Recall, request: Request):
            limit(request)
            out = mm.retrieve(user(body.session), body.query, agent_id="demo", limit=3)
            return {
                "status": out.get("status"),
                "answer": (out.get("answer") or {}).get("text"),
                "memories": [{"content": m.get("content"), "answer": m.get("answer")} for m in (out.get("memories") or [])[:3]],
                "reason": (out.get("receipt") or {}).get("reason"),
                "latency_ms": out.get("latency_ms"),
            }

        return api
