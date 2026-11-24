"""
Learned relevance model for recall.

    P(relevant | query, memory) = sigmoid(bias + sum_k w_k * x_k)

Features (all in [0, 1]) for one candidate memory:
  semantic          cosine(query embedding, memory embedding), clipped to [0, 1]
  lexical           BM25 magnitude / max BM25 magnitude among this query's candidates
  entity_index      memory was found through the entity index (its fact is about a named query entity)
  entity_match      memory is about a named entity of the question (owner, non-vocative mention, or fact slot)
  entity_conflict   question names entities; memory only names other ones
  subject_match     the memory is a first-person statement by an entity the question asks about
  subject_conflict  the memory is a first-person statement by someone else, and the question's
                    entities appear in it at most as direct address ("Thanks, Caroline!")
  relation_match    one of the memory's fact slots has the relation asked about
  relation_text     no such slot, but the memory text uses a word of that relation group
  relation_conflict question asks a relation; memory has fact slots, none of them that relation
  slot_match        a fact slot matches both the entity and the relation asked about
  salience          importance / 10
  recency           rank of the memory's timestamp among the candidates (1 = newest)
  alias_bridge      memory only defines an alias used to expand the query
  own_private       memory is private to the calling agent (vs shared)

Weights and the abstention threshold are fitted (fit_ranker.py) on labelled
recall data — the eval dev split, and/or real feedback logged through
POST /memory/feedback. A fitted model in <storage>/ranker_model.json
overrides the defaults below.
"""
import json
import math
import os
import re
import sqlite3
import time

from backend.utils.paths import get_db_path

FEATURES = ["semantic", "lexical", "entity_index", "entity_match", "entity_conflict", "subject_match",
            "subject_conflict", "relation_match", "relation_text", "relation_conflict", "slot_match", "salience",
            "recency", "alias_bridge", "own_private"]

# Shipped default model, produced by `python eval/run_eval.py --fit-ranker` (provenance
# is recorded inside the file). A deployment-specific fit in <storage>/ranker_model.json
# (fit_ranker.py --from-db) takes precedence.
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ranker_model.default.json")


PRIOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ranker_model.prior.json")


def _complete(model):
    """Fill weights for features the model predates from the documented prior (never crash on new features)."""
    with open(PRIOR_MODEL_PATH) as f:
        prior = json.load(f)["weights"]
    missing = [k for k in ["bias"] + FEATURES if k not in model.get("weights", {})]
    if missing:
        model = dict(model, weights={**prior, **model.get("weights", {})}, missing_from_fit=missing)
    return model


def _load_default():
    with open(DEFAULT_MODEL_PATH) as f:
        return _complete(json.load(f))


DEFAULT_MODEL = _load_default()


def model_path():
    return get_db_path("backend/storage/ranker_model.json")


class Ranker:
    _model = None
    _mtime = None

    @classmethod
    def model(cls):
        path = model_path()
        mtime = (path, os.path.getmtime(path) if os.path.exists(path) else None)
        if cls._model is None or mtime != cls._mtime:
            model = DEFAULT_MODEL
            if mtime[1] is not None:
                try:
                    with open(path) as f:
                        loaded = json.load(f)
                    if "threshold" in loaded and loaded.get("weights"):
                        model = _complete(loaded)
                except (OSError, ValueError):
                    pass
            cls._model, cls._mtime = model, mtime
        return cls._model

    @classmethod
    def score(cls, features, model=None):
        import math
        w = (model or cls.model())["weights"]
        z = w["bias"] + sum(w[k] * features.get(k, 0.0) for k in FEATURES)
        z = max(-35.0, min(35.0, z))  # numerically stable sigmoid
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def threshold(cls, model=None):
        return float((model or cls.model())["threshold"])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def memory_slots(memory):
    metadata = memory.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}
    slots = metadata.get("slots", []) if isinstance(metadata, dict) else []
    return slots if isinstance(slots, list) else []


def memory_speaker(memory):
    """Speaker recorded by the client at store time (metadata.client.speaker), if any."""
    metadata = memory.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}
    client = metadata.get("client") if isinstance(metadata, dict) else None
    speaker = client.get("speaker") if isinstance(client, dict) else None
    return speaker if isinstance(speaker, str) and speaker.strip() else None


def candidate_features(candidates, analysis, agent_id, alias_sources=()):
    """Attaches c["features"] to every candidate (needs c["semantic_score"] set)."""
    from backend.brain.features.query import (entity_matches, spans_match, relation_matches, slot_matches,
                                              memory_subject, RELATION_GROUPS)
    q_spans = analysis.get("strong_spans", [])
    asks_rel = analysis["asks_relation"]
    q_rels = analysis["relations"]
    asked_words = set().union(*(g for g in RELATION_GROUPS if g & q_rels)) if asks_rel else set()
    asks_name = bool(q_rels & {"name", "called", "alias"})

    max_bm25 = max((abs(c.get("bm25_score") or 0.0) for c in candidates), default=0.0)
    order = sorted(range(len(candidates)), key=lambda i: float(candidates[i].get("timestamp") or 0.0))
    recency = {i: (rank / (len(order) - 1) if len(order) > 1 else 1.0) for rank, i in enumerate(order)}

    for i, c in enumerate(candidates):
        slots = memory_slots(c)
        content = c.get("content", "")
        owner, mentions = memory_subject(content, memory_speaker(c))
        about = list(mentions) + ([owner] if owner else [])
        slot_ents = [s.get("entity", "") for s in slots if s.get("entity")]

        def is_about(q):
            return any(spans_match(q, m) for m in about) or any(entity_matches("".join(q), e) for e in slot_ents)

        ent_match = bool(q_spans) and any(is_about(q) for q in q_spans)
        subject_match = bool(owner) and any(spans_match(q, owner) for q in q_spans)
        mentioned = any(spans_match(q, m) for q in q_spans for m in mentions) or \
            any(entity_matches("".join(q), e) for q in q_spans for e in slot_ents)
        subject_conflict = bool(owner) and bool(q_spans) and not subject_match and not mentioned
        named = [m for m in about if m != ("user",)]
        rel_match = asks_rel and any(relation_matches(s.get("relation", ""), q_rels) for s in slots)
        words = set(re.findall(r"[a-z]+", content.lower()))
        bm25 = abs(c.get("bm25_score") or 0.0)
        c["features"] = {
            "semantic": min(max(float(c.get("semantic_score") or 0.0), 0.0), 1.0),
            "lexical": (bm25 / max_bm25) if max_bm25 > 0 else 0.0,
            "entity_index": 1.0 if c.get("symbolic_match") else 0.0,
            "entity_match": 1.0 if ent_match else 0.0,
            "entity_conflict": 1.0 if (q_spans and not ent_match and named) else 0.0,
            "subject_match": 1.0 if subject_match else 0.0,
            "subject_conflict": 1.0 if subject_conflict else 0.0,
            "relation_match": 1.0 if rel_match else 0.0,
            "relation_text": 1.0 if (asks_rel and not slots and words & asked_words) else 0.0,
            "relation_conflict": 1.0 if (asks_rel and slots and not rel_match) else 0.0,
            "slot_match": 1.0 if any(slot_matches(s, analysis) for s in slots) else 0.0,
            "salience": min(max(float(c.get("importance") or 5.0) / 10.0, 0.0), 1.0),
            "recency": recency[i],
            "alias_bridge": 1.0 if (c.get("id") in alias_sources and not asks_name) else 0.0,
            "own_private": 1.0 if (c.get("memory_type") == "private" and c.get("agent_id") == agent_id) else 0.0,
        }
    return candidates


# ---------------------------------------------------------------------------
# Recall log (training data from real usage)
# ---------------------------------------------------------------------------
class RecallLog:
    """
    Stores the features of the top candidates of each recall so that explicit
    feedback (POST /memory/feedback) turns them into labelled training data.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/sqlite_db/recall_log.db")
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recall_candidates (
                recall_id TEXT, user_id TEXT, memory_id INTEGER, rank INTEGER,
                features TEXT, label INTEGER, timestamp REAL,
                PRIMARY KEY (recall_id, memory_id)
            )
        ''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recall_user ON recall_candidates(user_id, recall_id)")
        # Answer-check margin of each recall and the memories it was computed on: feedback that
        # marks one of those memories relevant turns the margin into calibration data for the
        # wrong-refusal guarantee (backend/tools/calibrate.py).
        # Recall-time distillation: the question of each recall, and confirmed (question -> memory)
        # pairs written when feedback marks a memory as the answer (RecallEngine.confirm).
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recall_queries (
                recall_id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT, query TEXT, timestamp REAL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS confirmed_answers (
                user_id TEXT, agent_id TEXT, query TEXT, embedding BLOB, embedding_model TEXT,
                memory_id INTEGER, recall_id TEXT, timestamp REAL,
                PRIMARY KEY (user_id, query, memory_id)
            )
        ''')
        # Trust log-odds of each recall's top memory: feedback on that recall labels it, which
        # recalibrates the trust score for this deployment (backend/tools/calibrate_trust.py).
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recall_trust (
                recall_id TEXT PRIMARY KEY, user_id TEXT, memory_id INTEGER, logit REAL, timestamp REAL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recall_answer_check (
                recall_id TEXT PRIMARY KEY, user_id TEXT, model TEXT, best_margin REAL,
                checked_ids TEXT, timestamp REAL
            )
        ''')
        # What each feedback call said, verbatim: the memories marked used ([] = none was).
        # Learning the answer-check threshold needs "none used" as a signal of its own
        # (backend/brain/features/answer_check_learning.py).
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recall_feedback (
                recall_id TEXT PRIMARY KEY, user_id TEXT, used_ids TEXT, timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()

    def log(self, recall_id, user_id, ranked, top_n=20):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO recall_candidates VALUES (?, ?, ?, ?, ?, NULL, ?)",
                [(recall_id, user_id, c["id"], rank, json.dumps(c.get("features", {})), time.time())
                 for rank, c in enumerate(ranked[:top_n]) if c.get("features")])
        conn.close()

    def log_answer_check(self, recall_id, user_id, model, best_margin, checked_ids):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("INSERT OR REPLACE INTO recall_answer_check VALUES (?, ?, ?, ?, ?, ?)",
                         (recall_id, user_id, model, float(best_margin), json.dumps([int(i) for i in checked_ids]),
                          time.time()))
        conn.close()

    def log_trust(self, recall_id, user_id, memory_id, logit):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("INSERT OR REPLACE INTO recall_trust VALUES (?, ?, ?, ?, ?)",
                         (recall_id, user_id, int(memory_id), float(logit), time.time()))
        conn.close()

    def trust_samples(self):
        """[(logit, label)] for recalls whose top memory received feedback (1 = confirmed as the answer)."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute(
            "SELECT t.logit, c.label FROM recall_trust t JOIN recall_candidates c "
            "ON c.recall_id = t.recall_id AND c.memory_id = t.memory_id WHERE c.label IS NOT NULL").fetchall()
        conn.close()
        return [(float(z), int(y)) for z, y in rows]

    def forget_memory(self, user_id, memory_id, count_only=False):
        """Removes (or counts) every log row that refers to a memory: {"confirmed_answers", "ranking_features", "trust_log"}."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:
                counts = {
                    "confirmed_answers": conn.execute("SELECT COUNT(*) FROM confirmed_answers WHERE user_id = ? AND memory_id = ?",
                                                      (user_id, memory_id)).fetchone()[0],
                    "ranking_features": conn.execute("SELECT COUNT(*) FROM recall_candidates WHERE user_id = ? AND memory_id = ?",
                                                     (user_id, memory_id)).fetchone()[0],
                    "trust_log": conn.execute("SELECT COUNT(*) FROM recall_trust WHERE user_id = ? AND memory_id = ?",
                                              (user_id, memory_id)).fetchone()[0],
                }
                if not count_only:
                    conn.execute("DELETE FROM confirmed_answers WHERE user_id = ? AND memory_id = ?", (user_id, memory_id))
                    conn.execute("DELETE FROM recall_candidates WHERE user_id = ? AND memory_id = ?", (user_id, memory_id))
                    conn.execute("DELETE FROM recall_trust WHERE user_id = ? AND memory_id = ?", (user_id, memory_id))
            return counts
        finally:
            conn.close()

    def log_query(self, recall_id, user_id, agent_id, query):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("INSERT OR REPLACE INTO recall_queries VALUES (?, ?, ?, ?, ?)",
                         (recall_id, user_id, agent_id, query, time.time()))
        conn.close()

    def query_of(self, recall_id, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        row = conn.execute("SELECT query, agent_id FROM recall_queries WHERE recall_id = ? AND user_id = ?",
                           (recall_id, user_id)).fetchone()
        conn.close()
        return row

    def add_confirmed(self, user_id, agent_id, query, vector, model, memory_ids, recall_id):
        import numpy as np
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.executemany("INSERT OR REPLACE INTO confirmed_answers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             [(user_id, agent_id, query, np.asarray(vector, dtype=np.float32).tobytes(), model,
                               int(mid), recall_id, time.time()) for mid in memory_ids])
        conn.close()

    def confirmed(self, user_id, model):
        """[(query, vector, memory_id)] confirmed for this user with embedding `model`."""
        import numpy as np
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute("SELECT query, embedding, memory_id FROM confirmed_answers WHERE user_id = ? "
                            "AND embedding_model = ?", (user_id, model)).fetchall()
        conn.close()
        return [(q, np.frombuffer(b, dtype=np.float32), mid) for q, b, mid in rows]

    def answer_check_feedback(self, model):
        """
        [(recall_id, margin, answerable)] for recalls that received feedback, from what the feedback said:
        answerable = a memory the answer check read was marked used; not answerable = none was used.
        Recalls where only a memory the check did not read was used are left out (a ranking miss,
        not a decision of the answer check).
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute("SELECT a.recall_id, a.best_margin, a.checked_ids, f.used_ids FROM recall_answer_check a "
                            "JOIN recall_feedback f ON f.recall_id = a.recall_id WHERE a.model = ?", (model,)).fetchall()
        conn.close()
        out = []
        for rid, margin, checked, used in rows:
            used, checked = set(json.loads(used or "[]")), set(json.loads(checked or "[]"))
            if not used:
                out.append((rid, float(margin), False))
            elif used & checked:
                out.append((rid, float(margin), True))
        return out

    def answerable_margins(self, model):
        """
        Margins of recalls where feedback marked one of the checked memories relevant: the
        question was answerable and its evidence was in front of the reader. Calibration data.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute(
            "SELECT a.best_margin, a.checked_ids, a.recall_id FROM recall_answer_check a WHERE a.model = ?", (model,)).fetchall()
        out = []
        for margin, checked, rid in rows:
            ids = set(json.loads(checked or "[]"))
            hit = conn.execute("SELECT 1 FROM recall_candidates WHERE recall_id = ? AND label = 1 AND memory_id IN (%s) LIMIT 1"
                               % ",".join("?" * len(ids)), (rid, *ids)).fetchone() if ids else None
            if hit:
                out.append(margin)
        conn.close()
        return out

    def feedback(self, recall_id, user_id, relevant_ids):
        """Marks the listed memories relevant (1) and the other logged candidates not relevant (0)."""
        relevant = {int(i) for i in relevant_ids}
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            rows = conn.execute("SELECT memory_id FROM recall_candidates WHERE recall_id = ? AND user_id = ?",
                                (recall_id, user_id)).fetchall()
            if rows:  # only for this user's own recalls
                conn.execute("INSERT OR REPLACE INTO recall_feedback VALUES (?, ?, ?, ?)",
                             (recall_id, user_id, json.dumps(sorted(relevant)), time.time()))
            for (mid,) in rows:
                conn.execute("UPDATE recall_candidates SET label = ? WHERE recall_id = ? AND memory_id = ?",
                             (1 if mid in relevant else 0, recall_id, mid))
        conn.close()
        return len(rows)

    def labelled(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute("SELECT recall_id, features, label FROM recall_candidates WHERE label IS NOT NULL").fetchall()
        conn.close()
        return [(rid, json.loads(f), int(lbl)) for rid, f, lbl in rows]

    def clear_user(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM recall_candidates WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM recall_answer_check WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM recall_queries WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM confirmed_answers WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM recall_trust WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM recall_candidates")
            conn.execute("DELETE FROM recall_answer_check")
            conn.execute("DELETE FROM recall_queries")
            conn.execute("DELETE FROM confirmed_answers")
            conn.execute("DELETE FROM recall_trust")
        conn.close()
