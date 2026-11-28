"""
Re-ranker: a small cross-encoder reads the question and each candidate memory *together* and
scores how well the memory answers it.

The first ranking compares the question and each memory separately (embeddings, word overlap,
fact slots). That misses links which only show when both are read at once: "What indoor
activities has Andrew done with his girlfriend?" vs "My GF and I volunteered at a pet shelter".
For questions whose answer is spread over several memories (multi-hop), the evidence is usually
among the candidates but ranked too low to be served.

The default model (cross-encoder/ms-marco-MiniLM-L-6-v2, 22M parameters, Apache 2.0) was
trained on MS MARCO search queries. No LLM. It runs on CPU.

It is slower than the rest of recall (measured on an Apple M4 CPU: median recall 140 ms -> about
600 ms at 50 candidates), so by default it runs only when a recall asks for it (deep=True, for
questions whose answer is spread over several memories); `enabled: true` runs it on every recall.

Configuration (config.yaml recall.rerank): enabled, model, candidates (how many of the first
ranking it reads), keep (results that stay exactly where they are), context (also read each
memory after the message before it), max_length, threads, device (null = GPU if available).
How it is applied is in recall_engine.RecallEngine._rerank.
"""
import os
import threading

from backend import settings


def config():
    c = settings.get("recall.rerank", {}) or {}
    enabled = bool(c.get("enabled", False))
    env = os.environ.get("RECALLIC_RERANK")  # "0"/"1" overrides config (CI, eval gate)
    if env is not None:
        enabled = env.strip() not in ("0", "false", "no", "")
    return {"enabled": enabled, "model": c.get("model") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidates": int(c.get("candidates", 50)), "keep": int(c.get("keep", 3)),
            "max_length": int(c.get("max_length", 256)), "threads": c.get("threads"), "device": c.get("device"),
            "preload": bool(c.get("preload", True)), "auto": bool(c.get("auto", False)),
            "context": bool(c.get("context", True)), "context_gap_minutes": float(c.get("context_gap_minutes", 60)),
            "context_max_words": None if c.get("context_max_words") is None else int(c["context_max_words"])}


class Reranker:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, model_name, max_length=256, threads=None, device=None):
        from sentence_transformers import CrossEncoder
        if threads:
            import torch
            torch.set_num_threads(int(threads))
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)  # None: GPU if available, else CPU
        self._infer_lock = threading.Lock()

    @classmethod
    def from_config(cls, force=False):
        """
        The configured re-ranker, or None if unavailable (logged once). Without `force`, also None
        when recall.rerank.enabled is off (it then runs only for recalls that ask for it: deep=True).
        """
        cfg = config()
        if not cfg["enabled"] and not force:
            return None
        key = (cfg["model"], cfg["max_length"], cfg["threads"], cfg["device"])
        with cls._lock:
            if key not in cls._instances:
                try:
                    cls._instances[key] = cls(cfg["model"], cfg["max_length"], cfg["threads"], cfg["device"])
                except Exception as e:  # missing sentence-transformers/torch, no network for the first download, ...
                    print(f"[Reranker] disabled: cannot load {cfg['model']}: {e}")
                    cls._instances[key] = None
            return cls._instances[key]

    def scores(self, query, texts):
        """One relevance score per text (higher = answers the question better)."""
        if not texts:
            return []
        with self._infer_lock:
            out = self.model.predict([(query, t or "") for t in texts], batch_size=32, show_progress_bar=False)
        return [float(x) for x in out]
