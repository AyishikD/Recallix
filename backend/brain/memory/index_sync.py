"""
Keeps the vector engine's in-memory indexes consistent with SQLite.

SQLite (episodic_events.embedding) is the source of truth. The engine holds one
index per user; it is lost when the engine restarts. sync():

  1. pushes the HNSW parameters from config.yaml to the engine;
  2. for every user, compares the engine's live count and build parameters
     with SQLite; rebuilds the namespace (drop + bulk_add) on any mismatch;
  3. drops engine namespaces that have no memories in SQLite.

It runs at API startup, whenever the client sees a new engine instance (a
restart), and from `python -m backend.tools.reindex`.
"""
import threading
import time

from backend import settings
from backend.utils.internal_client import engine_get, engine_post, on_instance_change, EngineError

_sync_lock = threading.Lock()


def sync(episodic, semantic, force=False, log=print):
    with _sync_lock:
        t0 = time.time()
        params = settings.hnsw_params()
        engine_post("/configure", params)
        model = semantic.model_name
        expected = episodic.embedding_counts(model)
        dims = episodic.embedding_dims(model)
        actual = engine_get("/namespaces")
        batch = int(settings.get("engine.bulk_batch_size", 1000))

        rebuilt, vectors = [], 0
        for user_id, count in expected.items():
            ns = actual.get(user_id)
            matches = (ns is not None and ns.get("live") == count and ns.get("dim") == dims.get(user_id)
                       and ns.get("M") == params["M"] and ns.get("efConstruction") == params["efConstruction"])
            if matches and not force:
                continue
            engine_post("/drop", {"namespace": user_id})
            for chunk in episodic.iter_embeddings(user_id, model, batch=batch):
                semantic.bulk_index(user_id, chunk)
                vectors += len(chunk)
            rebuilt.append(user_id)

        orphans = [ns for ns in actual if ns not in expected]
        for ns in orphans:
            engine_post("/drop", {"namespace": ns})

        report = {"users": len(expected), "rebuilt_users": len(rebuilt), "vectors_indexed": vectors,
                  "orphan_namespaces_dropped": len(orphans),
                  "memories_without_current_embedding": episodic.stale_embedding_count(model),
                  "seconds": round(time.time() - t0, 2)}
        log(f"[IndexSync] {report}")
        return report


def backfill(episodic, semantic, limit=None, log=print):
    """Embeds memories that have no embedding (or one from another model) and indexes them."""
    model = semantic.model_name
    done = failed = 0
    while limit is None or done + failed < limit:
        rows = episodic.rows_needing_embedding(model, limit=500)
        if not rows:
            break
        progressed = False
        for mid, user_id, content in rows:
            vec = semantic.embed(content)
            if vec is None:
                failed += 1
                continue
            episodic.set_embedding(mid, vec, model)
            semantic.index(user_id, mid, vec)
            done += 1
            progressed = True
        if not progressed:
            break  # embedding provider is failing; avoid an infinite loop
    log(f"[IndexSync] backfill: embedded {done}, failed {failed}")
    return {"embedded": done, "failed": failed}


_watching = {"registered": False}


def watch_engine_restarts(get_manager):
    """
    Rebuild in the background whenever the engine instance changes after first
    contact. Registers one listener per process; the manager is looked up at
    call time so it is always the current one.
    """
    if _watching["registered"]:
        return
    _watching["registered"] = True

    def on_change(old, new):
        if old is None:
            return  # first contact; startup sync handles it
        print(f"[IndexSync] vector engine restarted ({old} -> {new}); rebuilding indexes")

        def run():
            mm = get_manager()
            try:
                sync(mm.episodic, mm.semantic)
            except EngineError as e:
                print(f"[IndexSync] rebuild failed: {e}")
        threading.Thread(target=run, daemon=True).start()
    on_instance_change(on_change)
