"""
Recall pipeline (no LLM):

  0. alias expansion from stored "X is also known as Y" facts
  1. candidates: lexical (FTS5 + entity index) and vector (this user's HNSW
     namespace), both bounded by recall.candidate_pool and filtered by agent
     visibility in SQL
  2. scoring: learned logistic relevance model over interpretable features
     (memory_ranking/ranker.py)
  3. truth arbitration: newest value per fact slot (TruthEngine)
  4. multi-hop: facts about entities a top memory points to ("contains",
     "powered by", "depends on")
  5. abstention: if the best candidate's relevance is below the learned
     threshold, return nothing rather than a guess
"""
import json
import re
import sqlite3
import time
import uuid

from backend import settings
from ..features.query import (analyze_query, capitalised_entities, norm_id, entity_matches, slot_matches,
                              resolve_first_person)
from ..memory_ranking.ranker import Ranker, RecallLog, candidate_features, memory_slots, memory_speaker
from ..memory_ranking.reinforcement_learning import MemoryRL
from ..features.answer_check import AnswerChecker

_INTERNAL_KEYS = ("_query_slot", "_slot_id", "_slot_value", "features", "semantic_score", "bm25_score",
                  "symbolic_match", "pool")


class RecallEngine:
    def __init__(self, episodic, semantic):
        self.episodic = episodic
        self.semantic = semantic
        self.recall_log = RecallLog()
        self.rl = MemoryRL()

    @staticmethod
    def classify_query(analysis):
        """Reported as `intent`: identity / symbolic / conceptual."""
        if "user" in analysis["strong_entities"] and analysis["asks_relation"]:
            return "identity"
        if [e for e in analysis["strong_entities"] if e != "user"] or analysis["asks_relation"]:
            return "symbolic"
        return "conceptual"

    # ------------------------------------------------------------------
    @staticmethod
    def _replace_alias(query, name, thing):
        """
        Rewrite the words of `query` that spell the alias `name` (normalised id, optionally
        preceded by "the" and wrapped in quotes) as the canonical `thing`.
        """
        words = list(re.finditer(r"[\w'’-]+", query))
        for i in range(len(words)):
            for j in range(i, min(i + 6, len(words))):
                joined = re.sub(r"[^a-z0-9]+", "", query[words[i].start():words[j].end()].lower())
                if joined in (name, "the" + name):
                    start, end = words[i].start(), words[j].end()
                    while start > 0 and query[start - 1] in "'\"‘“":
                        start -= 1
                    while end < len(query) and query[end] in "'\"’”":
                        end += 1
                    return query[:start] + thing + query[end:]
        return None

    def _resolve_aliases(self, user_id, query, agent_id):
        """
        Aliases from stored "X is also known as Y" facts. A question using the alias is
        rewritten to the canonical name; a question using the canonical name is expanded
        with the alias. Returns (rewritten query, alias-definition memory ids).
        """
        expansions, sources = [], set()
        # same normalisation as stored slot entities (lowercase alphanumerics; "Dragon's" -> "dragons")
        q_norm = re.sub(r"[^a-z0-9]+", "", query.lower())
        rewritten = query
        pool = int(settings.get("recall.candidate_pool", 200))
        for m in self.episodic.search(user_id, query, agent_id=agent_id, limit=pool):
            for s in memory_slots(m):
                if s.get("relation") != "alias_of":
                    continue
                name, thing = s.get("entity", ""), s.get("value", "")
                thing_norm = re.sub(r"[^a-z0-9]+", "", thing.lower())
                if thing_norm and thing_norm in q_norm and name not in q_norm:
                    display = next((span for span in re.findall(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*", m.get("content", ""))
                                    if re.sub(r"[^a-z0-9]+", "", span.lower()) in (name, "the" + name)), name)
                    expansions.append(display)
                    sources.add(m["id"])
                elif name and name in q_norm and thing_norm not in q_norm:
                    replaced = self._replace_alias(rewritten, name, thing)
                    rewritten = replaced if replaced is not None else f"{rewritten} {thing}"
                    sources.add(m["id"])
        if expansions:
            rewritten = f"{rewritten} {' '.join(dict.fromkeys(expansions))}"
        return rewritten, sources

    def _generate_candidates(self, user_id, query, agent_id, q_vec):
        pool = int(settings.get("recall.candidate_pool", 200))
        lexical = self.episodic.search(user_id, query, agent_id=agent_id, limit=pool)
        candidates = {c["id"]: c for c in lexical}
        vector_hits = self.semantic.search(user_id, query, top_k=pool, vector=q_vec) if q_vec is not None else []
        missing = [mid for mid, _ in vector_hits if mid not in candidates]
        fetched = self.episodic.get_many(user_id, missing, agent_id=agent_id)  # one batched query
        for mid, score in vector_hits:
            if mid in candidates:
                candidates[mid]["semantic_score"] = score
            elif mid in fetched:
                fetched[mid]["semantic_score"] = score
                candidates[mid] = fetched[mid]
        out = list(candidates.values())
        self._attach_semantic_scores(q_vec, out)
        return out

    def _attach_semantic_scores(self, q_vec, candidates):
        """q · v from stored embeddings for lexical-only candidates; legacy rows are embedded once and saved."""
        pending = [c for c in candidates if c.get("semantic_score") is None]
        if not pending:
            return
        if q_vec is None:
            for c in pending:
                c["semantic_score"] = 0.0
            return
        model = self.semantic.model_name
        cached = self.episodic.get_embeddings([c["id"] for c in pending], model)
        budget = int(settings.get("recall.backfill_per_query", 50))
        for c in pending:
            vec = cached.get(c["id"])
            if vec is None and budget > 0:
                vec = self.semantic.embed(c.get("content", ""))
                if vec is not None:
                    self.episodic.set_embedding(c["id"], vec, model)
                    self.semantic.index(c["user_id"], c["id"], vec)
                    budget -= 1
            sim = self.semantic.cosine(q_vec, vec)
            c["semantic_score"] = 0.0 if sim is None else sim

    def _score(self, candidates, analysis, agent_id, alias_sources=()):
        candidate_features(candidates, analysis, agent_id, alias_sources)
        for c in candidates:
            c["relevance"] = Ranker.score(c["features"])
            # The fact slot that answers the question (TruthEngine arbitrates on it).
            q_slot = next((s for s in memory_slots(c) if slot_matches(s, analysis)), None)
            if q_slot:
                c["_query_slot"] = q_slot
        candidates.sort(key=lambda x: x["relevance"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    def multi_stage_recall(self, user_id, query, agent_id="default_agent", top_k=5, return_candidates=False):
        from .reasoning_engine import ReasoningEngine
        from .truth_engine import TruthEngine
        t0 = time.time()

        effective_query, alias_sources = self._resolve_aliases(user_id, query, agent_id)
        aliases = [effective_query] if effective_query != query else []
        analysis = analyze_query(effective_query)
        q_vec = self.semantic.embed(effective_query)

        candidates = self._generate_candidates(user_id, effective_query, agent_id, q_vec)
        ranked = self._score(candidates, analysis, agent_id, alias_sources)
        all_scored = list(ranked)

        truth = TruthEngine()
        resolved = truth.resolve(ranked)

        # Multi-hop: pull in facts about an entity the top memories point to;
        # rank them above the source only if they answer the asked relation better.
        sources = resolved[:2]
        hops = ReasoningEngine().identify_multi_hop_opportunities(sources, analysis)
        if hops and sources:
            src = sources[0]
            source_ids = {m["id"] for m in sources}
            known = {r["id"]: r for r in ranked}
            for hop_target in hops:
                target_id = norm_id(hop_target)
                hop_cands = self._generate_candidates(user_id, hop_target, agent_id, q_vec)
                for h in self._score(hop_cands, analysis, agent_id)[:20]:
                    if h["id"] in source_ids:
                        continue
                    mentions = any(entity_matches(target_id, e) for e in capitalised_entities(h.get("content", ""))) \
                        or any(entity_matches(target_id, s.get("entity", "")) for s in memory_slots(h))
                    if not mentions:
                        continue
                    better = h["features"]["relation_match"] > src["features"]["relation_match"]
                    target = known.setdefault(h["id"], h)
                    if target is h:
                        ranked.append(h)
                    target["is_hop_result"] = better
                    target["relevance"] = max(target["relevance"], src["relevance"] * (1.01 if better else 0.99))
            ranked.sort(key=lambda x: x["relevance"], reverse=True)
            resolved = truth.resolve(ranked)

        # Abstain when the best candidate's relevance (which includes the answer-level
        # attribution features) is below the learned threshold, or the configured override.
        override = settings.get("recall.abstain_threshold")
        threshold = float(override) if override is not None else Ranker.threshold()
        # One abstention mechanism at a time: the answer-level check reads the served
        # memories when it is available; otherwise the relevance threshold decides.
        answer_check = None
        checker = AnswerChecker.from_config()
        if not resolved:
            status, final = "NO_MATCH", []
        elif checker is not None:
            final = resolved[:top_k]
            # the reader sees first-person statements resolved to their recorded speaker
            checks = checker.check(query, [resolve_first_person(m.get("content", ""), memory_speaker(m)) for m in final])
            for m, c in zip(final, checks):
                m["answer"], m["answer_margin"] = c["answer"], round(c["margin"], 3)
            best = max(c["margin"] for c in checks)
            answer_check = {"model": checker.model_name, "threshold": checker.threshold(), "best_margin": round(best, 3)}
            status = "CONSISTENT" if best >= answer_check["threshold"] else "ABSTAINED"
            if status == "ABSTAINED":
                final = []
            elif (settings.get("recall.answer_check", {}) or {}).get("rerank", False):
                # memories in which the reader found an answer first (stable: keeps relevance order within)
                final = sorted(final, key=lambda m: m["answer_margin"] < answer_check["threshold"])
        elif resolved[0]["relevance"] < threshold:
            status, final = "ABSTAINED", []
        else:
            status, final = "CONSISTENT", resolved[:top_k]

        recall_id = uuid.uuid4().hex
        self.recall_log.log(recall_id, user_id, all_scored)
        self._reinforce(final)

        memories = []
        for m in final:
            out = {k: v for k, v in m.items() if k not in _INTERNAL_KEYS}
            out["explain"] = {k: round(v, 3) for k, v in m.get("features", {}).items()}
            out["relevance"] = round(m["relevance"], 4)
            if isinstance(out.get("metadata"), str):
                try:
                    out["metadata"] = json.loads(out["metadata"])
                except ValueError:
                    pass
            memories.append(out)

        result = {
            "recall_id": recall_id,
            "intent": self.classify_query(analysis),
            "status": status,
            "memories": memories,
            "debug": {
                "aliases": aliases,
                "query_entities": analysis["strong_entities"],
                "candidates": len(all_scored),
                "threshold": round(threshold, 4),
                "top_relevance": round(resolved[0]["relevance"], 4) if resolved else None,
                "model": Ranker.model().get("source", ""),
                "answer_check": answer_check,
            },
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
        if return_candidates:
            result["_candidates"] = all_scored
        return result

    def _reinforce(self, memories):
        if not memories:
            return
        conn = sqlite3.connect(self.episodic.db_path, timeout=30)
        try:
            with conn:
                for m in memories:
                    row = conn.execute("SELECT reinforcement_score FROM episodic_events WHERE id = ?", (m["id"],)).fetchone()
                    if row:
                        conn.execute("UPDATE episodic_events SET reinforcement_score = ?, retrieval_count = retrieval_count + 1 "
                                     "WHERE id = ?", (self.rl.reinforce_positive(row[0] or 1.0), m["id"]))
        finally:
            conn.close()
