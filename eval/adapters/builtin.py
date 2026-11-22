"""Adapters for Recallix and the two classic baselines (vector search, BM25)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .base import MemorySystem

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval" / "benchmarks"))


class _RecallixShared:
    """One sandbox, one ingest and one recall per question, shared by the served and ranked views."""
    _instances = {}

    def __init__(self, embedder):
        from backend.tools.sandbox import Sandbox
        self.box = Sandbox(embedder=embedder).__enter__()
        self.mem_to_turn, self.last, self.refs = {}, None, 0

    @classmethod
    def get(cls, embedder):
        inst = cls._instances.get(embedder) or cls(embedder)
        cls._instances[embedder] = inst
        inst.refs += 1
        return inst

    def ingest(self, user, turns):
        if user in self.mem_to_turn:
            return
        m2t = defaultdict(list)
        with self.box.silenced():
            for turn_id, text, speaker, ts in turns:
                res = self.box.mm.store(user, text, client_metadata={"speaker": speaker} if speaker else None,
                                        timestamp=ts)
                m2t[res["memory_id"]].append(turn_id)
        self.mem_to_turn[user] = m2t

    def recall(self, user, query, k):
        key = (user, query, k)
        if self.last is None or self.last[0] != key:
            with self.box.silenced():
                self.last = (key, self.box.mm.retrieve(user, query, limit=k, return_candidates=True))
        return self.last[1]

    def release(self, embedder):
        self.refs -= 1
        if self.refs == 0:
            self.box.__exit__(None, None, None)
            self._instances.pop(embedder, None)


class Recallix(MemorySystem):
    """
    Recallix in a sandbox (temporary storage, in-process exact vector engine).
    mode="served": what Recallix returns (after the answer check; may decline).
    mode="ranked": its full ranking, ignoring the decline decision.
    Both views share one ingest and one recall per question.
    """

    def __init__(self, embedder="hash", mode="served"):
        from backend.brain.features.facts import EXTRACTOR_VERSION
        self.name = "recallix" if mode == "served" else "recallix-ranked"
        self.mode, self.embedder = mode, embedder
        self.shared = _RecallixShared.get(embedder)
        self.config = {"embedder": embedder, "mode": mode, "extractor": EXTRACTOR_VERSION, "llm_calls": 0}

    def add_conversation(self, user, turns):
        self.shared.ingest(user, turns)

    def search(self, user, query, k):
        out = self.shared.recall(user, query, k)
        m2t = self.shared.mem_to_turn.get(user, {})
        mems = sorted(out["_candidates"], key=lambda c: -c["relevance"]) if self.mode == "ranked" else out["memories"]
        return [t for m in mems for t in m2t.get(m["id"], [])][:k]

    def close(self):
        self.shared.release(self.embedder)


class Dense(MemorySystem):
    """Plain vector search over the turns with the same embedding model."""

    def __init__(self, embedder="hash"):
        from backend.tools.sandbox import make_embedder
        self.name, self.config = "vector-search", {"embedder": embedder}
        self.embed = make_embedder(embedder)
        self.index = {}

    def add_conversation(self, user, turns):
        vecs = np.array([self.embed(t) or np.zeros(1) for _, t, _, _ in turns], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        self.index[user] = ([tid for tid, *_ in turns], vecs)

    def search(self, user, query, k):
        ids, vecs = self.index[user]
        q = np.asarray(self.embed(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        return [ids[i] for i in np.argsort(-(vecs @ q))[:k]]


class BM25(MemorySystem):
    """Okapi BM25 over the turns (k1=1.2, b=0.75)."""

    def __init__(self):
        self.name, self.config = "bm25", {"k1": 1.2, "b": 0.75}
        self.index = {}

    def add_conversation(self, user, turns):
        import locomo
        self.index[user] = ([tid for tid, *_ in turns], locomo.BM25([t for _, t, _, _ in turns]))

    def search(self, user, query, k):
        ids, bm = self.index[user]
        return [ids[i] for i in bm.top(query, k)]
