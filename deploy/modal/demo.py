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

_HERE = pathlib.Path(__file__).resolve()
# the repo root on your machine (deploy/modal/demo.py -> repo); inside the container this file is /root/demo.py and
# the local paths below are not used, so any value works there
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
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

ALLOWED_ORIGINS: list = []   # browsers never call this directly: only the Cloudflare Worker does (server to server)
MAX_MESSAGES = 5           # per visitor: per session AND per IP, per 24 hours
MAX_CHARS = 400            # per message or question
PER_MINUTE = 20            # requests per visitor (IP) per minute
DAILY_REQUESTS = 3000      # all visitors together per UTC day, so nobody can run up the compute bill
DAY_ZERO = 1_767_225_600   # 2026-01-01 UTC: demo messages are dated one day apart from here


@app.cls(cpu=2.0, memory=2048, scaledown_window=300, timeout=60, max_containers=1,
         secrets=[modal.Secret.from_name("recallic-demo-gate")])  # GATE_SECRET, shared with the Cloudflare Worker
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
        ip_stores = collections.defaultdict(collections.deque)
        daily = {"day": "", "n": 0}
        lock = threading.Lock()
        mm = self.mm

        import hmac
        import os
        gate = os.environ["GATE_SECRET"]

        def gated(request: Request):
            # only the Cloudflare Worker at demo-api.recallic.com knows the secret: everything else is refused,
            # including wake-up pings, so nobody can keep this container running
            if not hmac.compare_digest(request.headers.get("x-demo-gate") or "", gate):
                raise HTTPException(403, "Use the demo at https://www.recallic.com/demo")

        def client_ip(request: Request) -> str:
            return request.headers.get("x-demo-client") or "?"  # the visitor's IP, as the Worker saw it

        def limit(request: Request):
            ip = client_ip(request)
            now = time.time()
            with lock:
                today = time.strftime("%Y-%m-%d", time.gmtime(now))
                if daily["day"] != today:
                    daily.update(day=today, n=0)
                if daily["n"] >= DAILY_REQUESTS:
                    raise HTTPException(429, "The demo has had a busy day. Please come back tomorrow.")
                daily["n"] += 1
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
        def health(request: Request):
            gated(request)
            return {"status": "ok"}

        @api.post("/store")
        def store(body: Store, request: Request):
            gated(request)
            limit(request)
            uid = user(body.session)
            ip = client_ip(request)
            now = time.time()
            with lock:
                q = ip_stores[ip]
                while q and now - q[0] > 86400:
                    q.popleft()
                if counts[uid] >= MAX_MESSAGES or len(q) >= MAX_MESSAGES:
                    raise HTTPException(429, f"The demo keeps {MAX_MESSAGES} messages per visitor. Ask about them, "
                                             "or come back tomorrow for a fresh start.")
                counts[uid] += 1
                q.append(now)
                day = counts[uid]
            # each message counts as one day after the previous one (a week of texting), so "the newest fact wins"
            # works as it does in real use, where messages arrive days apart rather than within one second
            ts = DAY_ZERO + day * 86400
            out = mm.store(uid, body.content, agent_id="demo", timestamp=ts)
            return {"status": out["status"], "memory_id": out["memory_id"], "ms": out.get("total_ms")}

        @api.post("/recall")
        def recall(body: Recall, request: Request):
            gated(request)
            limit(request)
            out = mm.retrieve(user(body.session), body.query, agent_id="demo", limit=3)
            best = out.get("answer") or {}
            first = (out.get("memories") or [{}])[0]
            # show the answer the receipt states: the first memory's (an update is ranked first). The top-level
            # `answer` can still hold an older, higher-confidence answer; see the note in the project memory.
            shown = first.get("answer") or best.get("later_answer") or best.get("text")
            return {
                "status": out.get("status"),
                "answer": shown,
                "replaced": best.get("text") if best.get("text") and best.get("text") != shown else None,
                "memories": [{"content": m.get("content"), "answer": m.get("answer")} for m in (out.get("memories") or [])[:3]],
                "reason": (out.get("receipt") or {}).get("reason"),
                "latency_ms": out.get("latency_ms"),
            }

        return api
