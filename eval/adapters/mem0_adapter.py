"""
Mem0 (open-source library) adapter — UNVERIFIED.

Written from Mem0's public documentation; we have not run it (it needs the `mem0ai`
package and, with its default configuration, an LLM API key, since Mem0 uses an LLM to
write memories). Results from it must not be published until it has been run and,
ideally, reviewed by Mem0's maintainers. Corrections by pull request are welcome — the
point of the harness is that every vendor's configuration is visible and fair.

Evidence mapping: each turn is added with metadata {"turn_id": ...}; Mem0 rewrites turns
into derived memories, and we map a search result back to evidence through that metadata.
If Mem0 merges several turns into one memory only one turn id may survive, which would
under-count its evidence recall — a known limitation of comparing at turn level.
"""
from .base import MemorySystem


class Mem0(MemorySystem):
    name = "mem0"

    def __init__(self, config=None):
        try:
            from mem0 import Memory
        except ImportError as e:
            raise SystemExit("the mem0 adapter needs `pip install mem0ai` (and its LLM/embedding credentials)") from e
        self.config = {"library": "mem0ai", "config": config or "default", "verified": False}
        self.memory = Memory.from_config(config) if config else Memory()

    def add_conversation(self, user, turns):
        for turn_id, text, speaker, ts in turns:
            self.memory.add([{"role": "user", "content": text}], user_id=user,
                            metadata={"turn_id": turn_id, "speaker": speaker})

    def search(self, user, query, k):
        res = self.memory.search(query, user_id=user, limit=k)
        rows = res.get("results", []) if isinstance(res, dict) else res
        out = []
        for r in rows:
            tid = (r.get("metadata") or {}).get("turn_id")
            if tid and tid not in out:
                out.append(tid)
        return out[:k]
