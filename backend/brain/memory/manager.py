import threading
import time

from backend import settings
from .recall_engine import RecallEngine
from .short_term.sensory import SensoryMemory
from .working.working_memory import WorkingMemory
from .episodic.episodic_memory import EpisodicMemory
from .semantic.semantic_memory import SemanticMemory
from .long_term.long_term_memory import LongTermMemory
from .reflective.reflective_memory import ReflectiveMemory
from .archive.archive_store import ArchiveStore
from ..prediction.predictive_recall import PredictiveRecall
from ..world_model.state_inference import StateInference
from ..world_model.timeline_engine import TimelineEngine
from ..world_model.prediction_engine import PredictionEngine
from ..world_model.planning_engine import PlanningEngine
from ..memory_ranking.importance_ranker import ImportanceRanker, content_terms
from ..features.facts import extract_triads, question_keys, resolve_definite, to_slots, with_learned
from ..features.learned_facts import LearnedExtractor
from .updates import CONTEXT_KEY
from ..features.schemas import SchemaClassifier
from ..features.implicit import ImplicitInference, rules_config as implicit_config

MEMORY_TYPES = ("private", "shared")


class MemoryManager:
    def __init__(self):
        self.sensory = SensoryMemory()
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.long_term = LongTermMemory()
        self.reflective = ReflectiveMemory()
        self.archive = ArchiveStore()
        self.schema_classifier = SchemaClassifier(self.semantic.embed, self.semantic.model_name)

        self.predictive = PredictiveRecall(self)
        self.state_inference = StateInference()
        self.timeline = TimelineEngine()
        self.prediction_engine = PredictionEngine()
        self.planning_engine = PlanningEngine(self)

        self.recall_engine = RecallEngine(self.episodic, self.semantic)
        from .audit import AuditLog
        self.audit = AuditLog()
        self._store_lock = threading.Lock()  # serialises dedupe-check + insert

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------
    def store(self, user_id, message, agent_id="default_agent", memory_type="private", client_metadata=None,
              timestamp=None, **_ignored):
        """
        Deterministic store pipeline (no LLM). Extra keyword arguments from
        older clients (skip_llm, sync_index) are accepted and ignored: writes
        are searchable immediately because the engine scans pending vectors.

        `timestamp` (seconds) records when the message was originally said; it defaults
        to now. Replay passes the original value so re-processing history reproduces
        every derived field (slot versions, truth order) exactly.
        """
        ts = time.time() if timestamp is None else float(timestamp)
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {MEMORY_TYPES}")
        logs = []
        t0 = time.time()

        def log(system, stage, detail, started):
            logs.append({"system": system, "stage": stage, "detail": detail,
                         "duration_ms": round((time.time() - started) * 1000, 1)})

        s = time.time()
        duplicate_of = self.episodic.find_duplicate(user_id, message, agent_id)
        if duplicate_of is not None:
            log("python", "Deduplication", f"Identical to memory {duplicate_of}", s)
            return {"status": "skipped_duplicate", "memory_id": duplicate_of, "importance": None,
                    "schemas": [], "promoted": False, "total_ms": round((time.time() - t0) * 1000, 1),
                    "processing_log": logs}

        s = time.time()
        self.sensory.add(user_id, message)
        before = self._context_before(user_id, message)  # recall.neighbour_context (read before the window moves)
        self.working.update(user_id, message)
        log("python", "Sensory & Working Memory", "Updated sliding windows", s)

        s = time.time()
        vector = self.semantic.embed(message)
        log("embedding", "Embedding", "OK" if vector is not None else "Unavailable (lexical only)", s)

        s = time.time()
        metadata_slots = resolve_definite(
            to_slots(with_learned(message, extract_triads(message), LearnedExtractor.from_config()), timestamp=ts),
            lambda rel: self.episodic.latest_entity_for_relation(user_id, rel, agent_id, ts))
        log("python", "Fact Extraction", f"Slots: {[m['slot_id'] for m in metadata_slots]}", s)

        conflicts = self._conflicts(user_id, agent_id, metadata_slots, ts)
        if conflicts:
            log("python", "Conflict Check", f"{len(conflicts)} fact(s) differ from what was said before", time.time())

        implied = []
        inference = ImplicitInference.from_config()
        if inference is not None:
            s = time.time()
            cfg = implicit_config()
            implied = inference.infer(message, (client_metadata or {}).get("speaker"),
                                      min_entailment=cfg["store_min_entailment"],
                                      min_similarity=cfg["store_pattern_similarity"])
            log("python", "Implicit Facts", f"{len(implied)} verified implications", s)

        s = time.time()
        schema_tags = self.schema_classifier.classify(message, vector)
        log("python", "Schema Classification", f"Schemas: {schema_tags}", s)

        s = time.time()
        max_sim = self.semantic.nearest_similarity(user_id, vector)
        total_docs, doc_freqs = self.episodic.term_stats(content_terms(message))
        importance, features = ImportanceRanker.calculate(message, max_sim, doc_freqs, total_docs)
        log("python", "Importance Scoring",
            f"Score: {importance:.1f}/10 | " + ", ".join(f"{k}={v:.2f}" for k, v in features.items()), s)

        event = {
            "user_id": user_id,
            "agent_id": agent_id,
            "memory_type": memory_type,
            "content": message,
            "timestamp": ts,
            "importance": importance,
            "embedding": vector,
            "embedding_model": self.semantic.model_name,
            # kept so fit_importance.py can learn weights from real recall usage
            "metadata": {"schema_tags": schema_tags, "slots": metadata_slots,
                         **({"implied": implied} if implied else {}),
                         "importance_features": {k: round(v, 4) for k, v in features.items()},
                         **({"client": client_metadata} if client_metadata else {})},
        }

        s = time.time()
        with self._store_lock:
            duplicate_of = self.episodic.find_duplicate(user_id, message, agent_id)
            if duplicate_of is not None:  # a concurrent request stored it meanwhile
                return {"status": "skipped_duplicate", "memory_id": duplicate_of, "importance": None,
                        "schemas": [], "promoted": False, "total_ms": round((time.time() - t0) * 1000, 1),
                        "processing_log": logs}
            memory_id = self.episodic.store(event)
        event["id"] = memory_id
        self.timeline.append_event(user_id, message, memory_id=memory_id, timestamp=event["timestamp"])
        log("python", "Episodic Storage", f"Stored in SQLite id={memory_id}", s)

        if metadata_slots and (settings.get("recall.question_keys", {}) or {}).get("enabled", False):
            s = time.time()
            questions = list(dict.fromkeys(q for slot in metadata_slots for q in question_keys(slot)))
            self.episodic.add_question_keys(memory_id, user_id, [(q, self.semantic.embed(q)) for q in questions],
                                            self.semantic.model_name)
            log("python", "Question Keys", f"{len(questions)} questions this memory answers", s)

        if before:
            s = time.time()
            text = f"{before} {message}"
            self.episodic.add_question_keys(memory_id, user_id, [(CONTEXT_KEY + text, self.semantic.embed(text))],
                                            self.semantic.model_name)
            log("embedding", "Neighbour Context", "Also searchable together with the message before it", s)

        s = time.time()
        indexed = self.semantic.index(user_id, memory_id, vector)
        log("cpp", "Vector Engine", "Indexed" if indexed else "Not indexed (no embedding / engine unavailable)", s)

        promoted = importance >= float(settings.get("memory.long_term_min_importance", 8.0))
        if promoted:
            s = time.time()
            self.long_term.promote(user_id, event)
            log("python", "Long-Term Promotion", f"importance {importance:.1f}", s)

        return {"status": "stored", "memory_id": memory_id, "importance": importance, "schemas": schema_tags,
                "promoted": promoted, "indexed": indexed, "conflicts": conflicts,
                "total_ms": round((time.time() - t0) * 1000, 1), "processing_log": logs}

    def _context_before(self, user_id, message):
        """
        recall.neighbour_context: a short message often only means something next to the one before
        it ("saw doc miller" after "shoulder hurts bad"; "Yes, in June" after a question). Such a
        message gets a second search key, the previous message(s) plus itself, so a question about
        the injury can find the doctor. Returns the text to put before it, or "" (off, long message,
        or nothing said before). The memory's own text, facts and vector are unchanged.
        """
        cfg = settings.get("recall.neighbour_context", {}) or {}
        if not cfg.get("enabled", False) or len(message.split()) > int(cfg.get("max_words", 12)):
            return ""
        recent = [m for m in (self.working.get(user_id) or {}).get("history", []) if isinstance(m, str) and m.strip()]
        return " ".join(recent[-int(cfg.get("previous", 1)):])

    _NO_CONFLICT = ("alias_of",)  # relations where a second value is an addition, not a contradiction

    def _conflicts(self, user_id, agent_id, slots, ts):
        """
        Facts in the new message whose value differs from the latest earlier value this agent can see:
        [{"fact", "previous_value", "previous_memory_id", "previous_said_at", "new_value"}]. The newest
        value still wins (TruthEngine); this tells the caller so it can confirm with the user.
        """
        candidates = [s for s in slots if s.get("relation") not in self._NO_CONFLICT]
        previous = self.episodic.latest_fact_values(user_id, [s["slot_id"] for s in candidates], agent_id, ts)
        norm = lambda v: " ".join(str(v or "").lower().split())  # noqa: E731
        out = []
        for s in candidates:
            prev = previous.get(s["slot_id"])
            if prev and norm(prev[0]) != norm(s["value"]):
                out.append({"fact": s["slot_id"], "previous_value": prev[0], "previous_memory_id": prev[1],
                            "previous_said_at": prev[2], "new_value": s["value"]})
        return out

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def proactive(self, user_id, message, agent_id="default_agent", limit=5):
        """Memories worth keeping in mind for `message`, with the reason (see memory/proactive.py)."""
        from .proactive import relevant
        return relevant(self.episodic, user_id, message, agent_id, limit)

    def retrieve(self, user_id, query, agent_id="default_agent", limit=5, return_candidates=False):
        return self.recall_engine.multi_stage_recall(user_id, query, agent_id=agent_id, top_k=limit,
                                                     return_candidates=return_candidates)

    # ------------------------------------------------------------------
    # Deletion (removes every derived copy)
    # ------------------------------------------------------------------
    def delete(self, user_id, memory_id, reason="user request"):
        """
        Deletes one memory and everything derived from it, checks that nothing is left, and records
        a receipt in the tamper-evident deletion log. Returns the receipt, or None if not found.
        """
        import sqlite3
        from .audit import deletion_receipt
        mid = int(memory_id)
        found = self.episodic.get_many(user_id, [mid])
        if mid not in found:
            return None
        content = found[mid].get("content", "")
        vector = self.episodic.get_embeddings([mid], self.semantic.model_name).get(mid)

        def counts():
            out = {}
            with sqlite3.connect(self.episodic.db_path) as c:
                out["memory"] = c.execute("SELECT COUNT(*) FROM episodic_events WHERE id = ? AND user_id = ?", (mid, user_id)).fetchone()[0]
                out["full_text_index"] = c.execute("SELECT COUNT(*) FROM episodic_fts WHERE rowid = ?", (mid,)).fetchone()[0]
                out["question_keys"] = c.execute("SELECT COUNT(*) FROM question_keys WHERE memory_id = ?", (mid,)).fetchone()[0]
            with sqlite3.connect(self.long_term.db_path) as c:
                out["long_term_copy"] = c.execute("SELECT COUNT(*) FROM long_term_facts WHERE user_id = ? AND source_id = ?", (user_id, mid)).fetchone()[0]
            with sqlite3.connect(self.timeline.db_path) as c:
                out["timeline_entry"] = c.execute("SELECT COUNT(*) FROM events WHERE user_id = ? AND memory_id = ?", (user_id, mid)).fetchone()[0]
            out.update(self.recall_engine.recall_log.forget_memory(user_id, mid, count_only=True))
            return out

        before = counts()
        self.episodic.delete(user_id, mid)
        vector_removed = self.semantic.remove(user_id, mid)
        self.long_term.delete_by_source(user_id, mid)
        self.timeline.delete_by_memory(user_id, mid)
        self.recall_engine.recall_log.forget_memory(user_id, mid)
        after = counts()
        removed = {k: v for k, v in before.items() if v}
        absent = {k: after[k] == 0 for k in before}
        removed["vector"] = int(bool(vector_removed))
        if vector is not None:  # the vector must no longer be the nearest match to itself
            hits = self.semantic.search(user_id, "", top_k=1, vector=vector)
            absent["vector"] = not any(h[0] == mid for h in hits)
        return self.audit.record(user_id, deletion_receipt(mid, content, removed, absent, reason))

    def clear_user(self, user_id):
        """Deletes all of one user's data in every store and their vector index."""
        n = self.episodic.clear_user(user_id)
        self.semantic.drop_user(user_id)
        for store in (self.long_term, self.reflective, self.timeline, self.state_inference,
                      self.working, self.sensory, self.archive, self.recall_engine.recall_log):
            store.clear_user(user_id)
        return n

    def clear_all(self):
        n = self.episodic.clear_all()
        self.semantic.drop_all()
        for store in (self.long_term, self.reflective, self.timeline, self.state_inference,
                      self.working, self.sensory, self.archive, self.recall_engine.recall_log):
            store.clear_all()
        return n
