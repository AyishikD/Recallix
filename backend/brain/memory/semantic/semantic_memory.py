"""
Embeddings and the vector index.

* Each user's vectors live in their own engine namespace (namespace = user_id,
  id = episodic memory id), so users never compete for top-k slots and one
  user's index can be dropped without touching anyone else's.
* Scores are raw cosine similarity; ranking happens in RecallEngine.
"""
import threading
from collections import OrderedDict

import numpy as np

from backend.brain.models.model_router import ModelRouter
from backend.utils.internal_client import engine_post, EngineError


class SemanticMemory:
    _CACHE_SIZE = 4096

    def __init__(self):
        self.router = ModelRouter()
        self.model_name = f"{self.router.embedding_provider}:{self.router.embedding_model}"
        self._cache = OrderedDict()  # text -> normalised float32 vector
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, text):
        """Unit-normalised float32 embedding, or None if the embedding model failed."""
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        raw = self.router.get_embedding(text)
        if not raw:
            return None  # never cache or index a failed embedding
        v = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-6:
            return None
        v = v / norm
        with self._lock:
            self._cache[text] = v
            if len(self._cache) > self._CACHE_SIZE:
                self._cache.popitem(last=False)
        return v

    @staticmethod
    def cosine(a, b):
        if a is None or b is None or len(a) != len(b):
            return None
        return float(np.dot(a, b))

    # ------------------------------------------------------------------
    # Vector index
    # ------------------------------------------------------------------
    def index(self, user_id, memory_id, vector):
        """Adds one memory's vector. Returns False (and logs) if the engine is unavailable."""
        if vector is None:
            return False
        payload = {"namespace": user_id, "id": str(memory_id), "vector": np.asarray(vector).tolist()}
        try:
            engine_post("/add", payload)
            return True
        except EngineError as e:
            if "dimension mismatch" in str(e):
                # The namespace holds vectors from a previous embedding model; they can
                # never match current queries. Start it over with the current model;
                # index_sync / backfill re-adds the user's other memories.
                try:
                    engine_post("/drop", {"namespace": user_id})
                    engine_post("/add", payload)
                    print(f"[SemanticMemory] reset index of {user_id} after an embedding model change")
                    return True
                except EngineError as e2:
                    e = e2
            print(f"[SemanticMemory] index failed (memory {memory_id} stays lexical-only until reindex): {e}")
            return False

    def bulk_index(self, user_id, items):
        """items: list of (memory_id, vector)."""
        if items:
            engine_post("/bulk_add", {"namespace": user_id,
                                      "items": [{"id": str(i), "vector": np.asarray(v).tolist()} for i, v in items]})

    def search(self, user_id, query, top_k=10, vector=None):
        """[(memory_id, cosine)] best first; [] if the engine is unavailable."""
        vector = self.embed(query) if vector is None else vector
        if vector is None:
            return []
        try:
            res = engine_post("/search", {"namespace": user_id, "query": vector.tolist(), "top_k": int(top_k)})
        except EngineError as e:
            print(f"[SemanticMemory] search unavailable, lexical recall only: {e}")
            return []
        out = []
        for r in res.get("results", []):
            try:
                out.append((int(r["id"]), float(r["score"])))
            except (KeyError, ValueError):
                continue
        return out

    def nearest_similarity(self, user_id, vector, top_k=5):
        """Highest cosine similarity to this user's indexed memories, or None if none/unavailable."""
        if vector is None:
            return None
        hits = self.search(user_id, "", top_k=top_k, vector=vector)
        return max((s for _, s in hits), default=None)

    def remove(self, user_id, memory_id):
        try:
            engine_post("/remove", {"namespace": user_id, "id": str(memory_id)})
            return True
        except EngineError as e:
            print(f"[SemanticMemory] remove failed for {memory_id}; it is filtered at recall and dropped on reindex: {e}")
            return False

    def drop_user(self, user_id):
        engine_post("/drop", {"namespace": user_id})

    def drop_all(self):
        engine_post("/drop_all", {})

    def compute_similarity(self, query, text):
        sim = self.cosine(self.embed(query), self.embed(text))
        return 0.5 if sim is None else sim
