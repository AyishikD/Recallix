"""
An isolated Recallix memory for tools: replay, memory tests, calibration, evaluation.

    with Sandbox(embedder="configured") as box:
        box.mm.store("u", "I live in Pune.")
        box.mm.retrieve("u", "Where do I live?")

Storage goes to a fresh temporary directory, and vectors to an in-process, exact,
namespaced engine (InProcessEngine), so a sandbox never touches the production
SQLite files or the running vector engine. Because the store pipeline's novelty
feature uses exact search, results match what the real engine would produce.

Embedders: "configured" (config.yaml models.embedding_provider), "hash" (fast,
deterministic, no model — for CI), or "st:<sentence-transformers model>".
"""
import contextlib
import hashlib
import io
import os
import shutil
import tempfile

import numpy as np


class InProcessEngine:
    """Exact, brute-force implementation of the vector engine's HTTP protocol."""

    def __init__(self):
        self.ns = {}
        self.params = {"M": 16, "efConstruction": 100, "efSearch": 50}

    def post(self, path, body=None):
        body = body or {}
        if path == "/configure":
            self.params.update({k: body[k] for k in ("M", "efConstruction", "efSearch") if k in body})
            return {"status": "ok"}
        if path in ("/add", "/bulk_add"):
            items = body["items"] if path == "/bulk_add" else [{"id": body["id"], "vector": body["vector"]}]
            space = self.ns.setdefault(body["namespace"], {})
            for it in items:
                v = np.asarray(it["vector"], dtype=np.float32)
                space[it["id"]] = v / (np.linalg.norm(v) or 1.0)
            return {"status": "ok", "added": len(items)}
        if path == "/search":  # always exact; "exact" and "ef_search" are accepted and need no handling
            space = self.ns.get(body["namespace"], {})
            if not space:
                return {"results": []}
            ids = list(space)
            q = np.asarray(body["query"], dtype=np.float32)
            sims = np.vstack([space[i] for i in ids]) @ (q / (np.linalg.norm(q) or 1.0))
            order = np.argsort(-sims, kind="stable")[: body.get("top_k", 10)]  # ties: insertion order
            return {"results": [{"id": ids[i], "score": float(sims[i])} for i in order]}
        if path == "/remove":
            return {"removed": self.ns.get(body["namespace"], {}).pop(body["id"], None) is not None}
        if path == "/drop":
            return {"dropped": self.ns.pop(body["namespace"], None) is not None}
        if path == "/drop_all":
            n = len(self.ns)
            self.ns.clear()
            return {"dropped": n}
        raise ValueError(path)

    def get(self, path):
        if path == "/namespaces":
            return {k: {"live": len(v), "M": self.params["M"], "efConstruction": self.params["efConstruction"],
                        "dim": len(next(iter(v.values()))) if v else 0} for k, v in self.ns.items()}
        if path == "/status":
            return {"status": "online", "instance": "in-process"}
        raise ValueError(path)


def hash_embedder(dim=384):
    """Deterministic bag-of-terms embedding: no model download, for CI and fast tests."""
    from backend.brain.memory_ranking.importance_ranker import content_terms

    def embed(text):
        v = np.zeros(dim)
        for t in content_terms(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1
        return v.tolist() if v.any() else []
    return embed


def make_embedder(spec):
    if spec == "hash":
        return hash_embedder()
    if spec.startswith("st:"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(spec[3:], cache_folder=os.environ.get("HF_CACHE"))
        return lambda text: model.encode([text])[0].tolist()
    raise SystemExit(f"unknown embedder {spec!r} (use configured, hash or st:<model>)")


class Sandbox:
    """Context manager: an isolated MemoryManager on temporary storage and an in-process engine."""

    def __init__(self, embedder="configured", answer_check=None, quiet=True):
        self.embedder, self.answer_check, self.quiet = embedder, answer_check, quiet
        self._saved = {}

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="recallix_sandbox_")
        for key, value in (("RECALLIX_STORAGE_ROOT", self.tmp),
                           ("RECALLIX_INTERNAL_KEY", os.environ.get("RECALLIX_INTERNAL_KEY") or "sandbox-not-a-secret")):
            self._saved[key] = os.environ.get(key)
            os.environ[key] = value
        if self.answer_check is not None:
            self._saved["RECALLIX_ANSWER_CHECK"] = os.environ.get("RECALLIX_ANSWER_CHECK")
            os.environ["RECALLIX_ANSWER_CHECK"] = "1" if self.answer_check else "0"
        from backend import settings
        settings.reload()

        import backend.brain.memory.semantic.semantic_memory as sm
        import backend.brain.memory.index_sync as isync
        import backend.brain.models.model_router as mr
        self.engine = InProcessEngine()
        self._patches = [(sm, "engine_post", sm.engine_post), (isync, "engine_post", isync.engine_post),
                         (isync, "engine_get", isync.engine_get)]
        sm.engine_post = self.engine.post
        isync.engine_post, isync.engine_get = self.engine.post, self.engine.get
        if self.embedder != "configured":
            raw = make_embedder(self.embedder)
            self._patches.append((mr.ModelRouter, "get_embedding", mr.ModelRouter.get_embedding))
            mr.ModelRouter.get_embedding = lambda _self, text: raw(text)

        from backend.brain.memory.manager import MemoryManager
        with self.silenced():
            self.mm = MemoryManager()
        return self

    def silenced(self):
        return contextlib.redirect_stdout(io.StringIO()) if self.quiet else contextlib.nullcontext()

    def __exit__(self, *exc):
        for obj, name, value in reversed(self._patches):
            setattr(obj, name, value)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from backend import settings
        settings.reload()
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False
