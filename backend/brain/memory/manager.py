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
from ..features.facts import extract_triads, question_keys, resolve_definite, to_slots
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
        self.working.update(user_id, message)
        log("python", "Sensory & Working Memory", "Updated sliding windows", s)

        s = time.time()
        vector = self.semantic.embed(message)
        log("embedding", "Embedding", "OK" if vector is not None else "Unavailable (lexical only)", s)

        s = time.time()
        metadata_slots = resolve_definite(
            to_slots(extract_triads(message), timestamp=ts),
            lambda rel: self.episodic.latest_entity_for_relation(user_id, rel, agent_id, ts))
        log("python", "Fact Extraction", f"Slots: {[m['slot_id'] for m in metadata_slots]}", s)

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

        s = time.time()
        indexed = self.semantic.index(user_id, memory_id, vector)
        log("cpp", "Vector Engine", "Indexed" if indexed else "Not indexed (no embedding / engine unavailable)", s)

        promoted = importance >= float(settings.get("memory.long_term_min_importance", 8.0))
        if promoted:
            s = time.time()
            self.long_term.promote(user_id, event)
            log("python", "Long-Term Promotion", f"importance {importance:.1f}", s)

        return {"status": "stored", "memory_id": memory_id, "importance": importance, "schemas": schema_tags,
                "promoted": promoted, "indexed": indexed, "total_ms": round((time.time() - t0) * 1000, 1),
                "processing_log": logs}

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def retrieve(self, user_id, query, agent_id="default_agent", limit=5, return_candidates=False):
        return self.recall_engine.multi_stage_recall(user_id, query, agent_id=agent_id, top_k=limit,
                                                     return_candidates=return_candidates)

    # ------------------------------------------------------------------
    # Deletion (removes every derived copy)
    # ------------------------------------------------------------------
    def delete(self, user_id, memory_id):
        """Deletes one memory and everything derived from it. Returns False if not found."""
        if not self.episodic.delete(user_id, memory_id):
            return False
        self.semantic.remove(user_id, memory_id)
        self.long_term.delete_by_source(user_id, memory_id)
        self.timeline.delete_by_memory(user_id, memory_id)
        return True

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
