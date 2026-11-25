"""
Recallix API server.

  python -m backend.tools.setup_env     # once: creates .env (engine secret) and an admin API key
  ./backend/infra_cpp/build/memory_service &   # vector engine (reads RECALLIX_* from the environment)
  python server.py                      # or: uvicorn server:app

Startup: verify the engine is reachable, push HNSW parameters, rebuild any
vector index that does not match SQLite, watch for engine restarts, start
background jobs.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import settings
from backend.api.agents.router import router as agents_router
from backend.api.deps import get_memory_manager, get_scheduler
from backend.api.internal.internal_api import router as internal_router
from backend.api.middleware.body_limit import BodySizeLimit
from backend.api.public.public_api import router as public_router
from backend.brain.memory import index_sync
from backend.utils.internal_client import EngineError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.internal_key()  # fail fast if the engine secret is missing
    mm = get_memory_manager()
    try:
        index_sync.sync(mm.episodic, mm.semantic)
    except EngineError as e:
        print(f"[Startup] Vector engine unavailable ({e}). Recall runs lexical-only until it is reachable; "
              f"indexes are rebuilt automatically when it is.")
    index_sync.watch_engine_restarts(get_memory_manager)
    get_scheduler().start()
    yield
    get_scheduler().stop()


app = FastAPI(title="Recallix API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(BodySizeLimit)  # api.max_request_bytes / api.max_import_bytes
app.include_router(public_router)
app.include_router(agents_router)
app.include_router(internal_router)


@app.get("/")
def root():
    return {"service": "recallix", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("RECALLIX_API_HOST", "127.0.0.1"),
                port=int(os.environ.get("RECALLIX_API_PORT", "8000")))
