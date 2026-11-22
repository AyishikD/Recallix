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
from ..features.implicit import ImplicitInference, rules_config as implicit_config
from ..features.facts import EXTRACTOR_VERSION
from . import freshness
from ..features import trust

_INTERNAL_KEYS = ("_query_slot", "_slot_id", "_slot_value", "features", "semantic_score", "bm25_score",
                  "symbolic_match", "pool", "_answer_ok")


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
        self._add_confirmed_candidates(user_id, agent_id, q_vec, candidates)
        out = list(candidates.values())
        self._attach_semantic_scores(q_vec, out)
        self._attach_question_keys(q_vec, out)
        return out

    def _add_confirmed_candidates(self, user_id, agent_id, q_vec, candidates):
        """
        Recall-time distillation (recall.distillation): memories that feedback confirmed as the
        answer to a similar earlier question (cosine >= min_similarity) always join the candidate
        pool. Ranking is unchanged; this only stops a confirmed answer from being missed. The
        memory is re-read under this agent's visibility, so deleted or private memories never leak.
        """
        cfg = settings.get("recall.distillation", {}) or {}
        if q_vec is None or not cfg.get("enabled", True):
            return
        floor = float(cfg.get("min_similarity", 0.9))
        hits = {}
        for query, vec, mid in self.recall_log.confirmed(user_id, self.semantic.model_name):
            sim = self.semantic.cosine(q_vec, vec)
            if sim is not None and sim >= floor and sim > hits.get(mid, (-1.0, ""))[0]:
                hits[mid] = (sim, query)
        if not hits:
            return
        fetched = self.episodic.get_many(user_id, [m for m in hits if m not in candidates], agent_id=agent_id)
        for mid, (sim, query) in hits.items():
            c = candidates.get(mid) or fetched.get(mid)
            if c is not None:
                c["confirmed_for"] = query
                candidates[mid] = c

    def confirm(self, recall_id, user_id, relevant_ids):
        """
        Feedback: labels the recall's candidates (ranker training data) and distills the
        confirmed (question -> memory) pairs for future recalls. Returns labelled candidates.
        """
        n = self.recall_log.feedback(recall_id, user_id, relevant_ids)
        row = self.recall_log.query_of(recall_id, user_id)
        if n and row and relevant_ids:
            query, agent_id = row
            vec = self.semantic.embed(query)
            if vec is not None:
                self.recall_log.add_confirmed(user_id, agent_id, query, vec, self.semantic.model_name,
                                              relevant_ids, recall_id)
        return n

    def _attach_question_keys(self, q_vec, candidates):
        """
        recall.question_keys: a memory's meaning score becomes the better of question-vs-memory
        and question-vs-the-questions-it-answers (stored at write time from its facts).
        """
        if q_vec is None or not (settings.get("recall.question_keys", {}) or {}).get("enabled", False):
            return
        keys = self.episodic.get_question_keys([c["id"] for c in candidates], self.semantic.model_name)
        for c in candidates:
            best = None
            for text, vec in keys.get(c["id"], []):
                sim = self.semantic.cosine(q_vec, vec)
                if sim is not None and (best is None or sim > best[0]):
                    best = (sim, text)
            if best and best[0] > (c.get("semantic_score") or 0.0):
                c["semantic_score"] = best[0]
                c["matched_question"] = best[1]

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
        outsider = self._unknown_person(query, all_scored)
        if outsider:
            status, final = "ABSTAINED", []
        elif not resolved:
            status, final = "NO_MATCH", []
        elif checker is not None:
            final = resolved[:top_k]
            t = checker.threshold()
            accept, subject = self._subject_contrast(checker, query, all_scored)
            read, best, answered = self._read_until_answer(checker, query, final, t, accept)
            cert = checker.certificate() if hasattr(checker, "certificate") else None  # custom checkers may omit it
            answer_check = {"model": checker.model_name, "threshold": t, "best_margin": round(best, 3),
                            "read": len(read), "guarantee": cert["statement"] if cert else None,
                            "subject": subject,
                            "rejected_as_about_someone_else": [m["id"] for m in read if m.get("subject_contrast") is not None
                                                               and not m.get("_answer_ok")]}
            status = "CONSISTENT" if answered else "ABSTAINED"
            if status == "ABSTAINED":
                inferred = self._answer_by_inference(checker, query, resolved[:3], t)
                if inferred is not None:
                    status, answer_check["inferred_from"] = "CONSISTENT", inferred["id"]
                    final = [inferred] + [m for m in resolved[:top_k] if m["id"] != inferred["id"]][: top_k - 1]
            if status == "ABSTAINED":
                final = []
            elif (settings.get("recall.answer_check", {}) or {}).get("rerank", False):
                # memories in which the reader found an answer first (stable: keeps relevance order within)
                final = sorted(final, key=lambda m: not m.get("_answer_ok"))
        elif resolved[0]["relevance"] < threshold:
            status, final = "ABSTAINED", []
        else:
            status, final = "CONSISTENT", resolved[:top_k]

        recall_id = uuid.uuid4().hex
        self.recall_log.log(recall_id, user_id, all_scored)
        self.recall_log.log_query(recall_id, user_id, agent_id, query)
        if answer_check is not None:
            self.recall_log.log_answer_check(recall_id, user_id, answer_check["model"], answer_check["best_margin"],
                                             [m["id"] for m in read])
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
            out["receipt"] = self._memory_receipt(out)
            memories.append(out)
        fresh = settings.get("recall.freshness", {}) or {}
        if fresh.get("enabled", True):
            freshness.annotate([f for m in memories for f in m["receipt"]["facts"]], self.episodic.db_path,
                               user_id, agent_id, time.time(), float(fresh.get("prior_events", 1.0)),
                               float(fresh.get("prior_days", 365.0)))
        for m in memories:  # calibrated P(this memory answers), when a fitted trust model is installed
            m["trust"] = trust.score(m)

        result = {
            "recall_id": recall_id,
            "intent": self.classify_query(analysis),
            "status": status,
            "memories": memories,
            "receipt": self._recall_receipt(status, resolved, final, threshold, answer_check, outsider),
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
        if memories:
            result["receipt"]["trust"] = memories[0].get("trust")
            result["receipt"]["trust_calibration"] = trust.trust_calibration()
            z = trust.logit(memories[0])
            if z is not None:  # feedback on this recall later labels it: deployment calibration data
                self.recall_log.log_trust(recall_id, user_id, memories[0]["id"], z)
        if return_candidates:
            result["_candidates"] = all_scored
        return result

    _PERSON_IN_QUESTION = re.compile(
        r"\b(?:does|did|do|is|was|are|were|has|have|had|will|can|would)\s+([A-Z][a-z]+)\b|\b([A-Z][a-z]+)['’]s\b")
    _NOT_PEOPLE = {"I", "The", "What", "When", "Where", "Who", "Which", "How", "Why", "January", "February", "March",
                   "April", "May", "June", "July", "August", "September", "October", "November", "December", "Monday",
                   "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}

    def _unknown_person(self, query, candidates):
        """
        recall.unknown_person_check: a question about a named person ("Where does Morgan work?",
        "What is Quinn's dog?") whom no candidate memory mentions and who never spoke cannot be
        answered from memory — returning the user's own facts would misattribute them. Returns the
        name, or None. Only person-like wording triggers it (not "my favourite Netflix show").
        """
        if not (settings.get("recall.unknown_person_check", {}) or {}).get("enabled", True):
            return None
        names = {a or b for a, b in self._PERSON_IN_QUESTION.findall(query)} - self._NOT_PEOPLE
        if not names:
            return None
        known_text = " ".join(c.get("content", "") for c in candidates[:200]).lower()
        speakers = {(memory_speaker(c) or "").lower() for c in candidates[:200]}
        for name in sorted(names):
            if name.lower() not in speakers and not re.search(rf"\b{re.escape(name.lower())}\b", known_text):
                return name
        return None

    @staticmethod
    def _answer_by_inference(checker, query, memories, threshold):
        """
        Implicit facts (recall.implicit): when no memory answers directly, the top memories are
        expanded into verified implications (stored at write time, or inferred now) and the
        reader looks for the answer in those. Returns the memory whose implication answered, or None.
        """
        inference = ImplicitInference.from_config()
        if inference is None:
            return None
        cfg = implicit_config()
        for m in memories:
            meta = m.get("metadata")
            meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
            implied = meta.get("implied") or inference.infer(
                m.get("content", ""), memory_speaker(m), min_entailment=cfg["question_min_entailment"],
                min_similarity=cfg["pattern_similarity"])
            if not implied:
                continue
            checks = checker.check(query, [i["statement"] for i in implied])
            best = max(range(len(checks)), key=lambda k: checks[k]["margin"])
            if checks[best]["margin"] >= threshold:
                m["answer"], m["answer_margin"], m["_answer_ok"] = checks[best]["answer"], round(checks[best]["margin"], 3), True
                m["inferred"] = {**implied[best], "source": inference.source}
                return m
        return None

    @staticmethod
    def _subject_contrast(checker, query, candidates):
        """
        "Who said what" check (config recall.answer_check.subject_contrast). If the question
        names exactly one of the people speaking in this user's memories, a memory only counts
        as answering it when the reader finds the answer *more strongly for that person* than
        when the same question is asked about each other speaker:

            contrast = margin(question) - max over other speakers s' of margin(question[subject -> s'])

        A memory that "answers" just as well for someone else is not evidence about the asked
        person. Accept when contrast > min_contrast (default 0: support the asked person more
        than anyone else; no fitting). Returns (accept(memory, context) or None, subject).
        """
        cfg = ((settings.get("recall.answer_check", {}) or {}).get("subject_contrast") or {})
        if not cfg.get("enabled", False):
            return None, None
        speakers = []
        for m in candidates[:200]:
            sp = memory_speaker(m)
            if sp and sp not in speakers:
                speakers.append(sp)
        named = [sp for sp in speakers if re.search(rf"\b{re.escape(sp)}\b", query, re.I)]
        if len(named) != 1 or len(speakers) < 2:
            return None, None
        subject = named[0]
        others = [sp for sp in speakers if sp != subject][: int(cfg.get("max_alternatives", 3))]
        swapped = [re.sub(rf"\b{re.escape(subject)}\b", alt, query, flags=re.I) for alt in others]
        min_contrast = float(cfg.get("min_contrast", 0.0))

        def accept(memory, context):
            alt = max(checker.check(q, [context])[0]["margin"] for q in swapped)
            memory["subject_contrast"] = round(memory["answer_margin"] - alt, 3)
            return memory["subject_contrast"] > min_contrast
        return accept, subject

    @staticmethod
    def _read_until_answer(checker, query, memories, threshold, accept=None):
        """
        Reads memories in rank order — 1, then 2 at a time — and stops at the first batch in
        which one contains an answer (margin >= threshold). The decision ("does any served
        memory answer?") is identical to reading all of them, but an answered recall usually
        needs one read instead of top_k. Declines still read everything. With a non-finite or
        calibration-mode threshold (<= -1e8) everything is read, so margins stay complete.
        `accept(memory, context)`, if given, must also hold for a memory to count as answering
        (the subject-contrast check). Returns (memories read, best margin, answered).
        """
        # the reader sees first-person statements resolved to their recorded speaker
        contexts = [resolve_first_person(m.get("content", ""), memory_speaker(m)) for m in memories]
        early_exit = threshold > -1e8
        read, best, answered, i = [], float("-inf"), False, 0
        while i < len(memories):
            size = (1 if i == 0 else 2) if early_exit else len(memories) - i
            batch, ctxs = memories[i:i + size], contexts[i:i + size]
            for m, c in zip(batch, checker.check(query, ctxs)):
                m["answer"], m["answer_margin"] = c["answer"], round(c["margin"], 3)
                best = max(best, c["margin"])
            for m, ctx in zip(batch, ctxs):
                if m["answer_margin"] >= threshold and (accept is None or accept(m, ctx)):
                    m["_answer_ok"] = True
                    answered = True
            read.extend(batch)
            i += size
            if early_exit and answered:
                break
        return read, best, answered

    @staticmethod
    def _memory_receipt(m):
        """Where a returned memory came from: who said it, when, and which rule extracted each fact."""
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        content = m.get("content", "")
        facts = []
        for slot in meta.get("slots", []) or []:
            src = slot.get("source") or {}
            span = src.get("span")
            facts.append({"fact": slot.get("slot_id"), "value": slot.get("value"), "rule": src.get("rule") or None,
                          "quote": content[span[0]:span[1]] if span else None, "extractor": src.get("extractor")})
        return {"memory_id": m.get("id"), "stored_at": m.get("timestamp"), "agent_id": m.get("agent_id"),
                "memory_type": m.get("memory_type"), "speaker": memory_speaker(m) or None, "facts": facts,
                "matched_question": m.get("matched_question"),
                "confirmed_for": m.get("confirmed_for"),
                "inferred": m.get("inferred"),
                "answer": m.get("answer") or None, "answer_margin": m.get("answer_margin")}

    def _recall_receipt(self, status, resolved, final, threshold, answer_check, outsider=None):
        """Why this recall answered or declined, and which models and rules were involved."""
        if outsider:
            reason = f"the question asks about {outsider}, who is not mentioned in any memory"
        elif status == "NO_MATCH":
            reason = "no stored memory matched the question"
        elif answer_check is not None:
            t, b = answer_check["threshold"], answer_check["best_margin"]
            rejected = answer_check.get("rejected_as_about_someone_else") or []
            src = next((m for m in final if m.get("inferred")), None) if answer_check.get("inferred_from") else None
            if src is not None:
                inf = src["inferred"]
                reason = (f"answer inferred from memory {src['id']}: \"{src.get('content', '')[:80]}\" implies "
                          f"\"{inf['statement']}\" (commonsense pattern \"{inf['pattern']}\" -> {inf['relation']}; "
                          f"entailment {inf['entailment']})")
            elif status == "ABSTAINED" and rejected:
                reason = (f"memories {rejected} answer a question like this, but about someone other than "
                          f"{answer_check.get('subject')}; nothing answers it about {answer_check.get('subject')}")
            elif status == "ABSTAINED":
                reason = f"none of the top memories contains an answer (best answer margin {b} < threshold {t:.3f})"
            else:
                top = next((m for m in final if m.get("_answer_ok")), final[0] if final else None)
                reason = (f"answer found in memory {top['id']}: \"{top.get('answer')}\" (margin {top.get('answer_margin')} "
                          f">= threshold {t:.3f})") if top else "answer found"
        elif status == "ABSTAINED":
            reason = f"best relevance {resolved[0]['relevance']:.3f} is below the threshold {threshold:.3f}"
        else:
            reason = f"best relevance {resolved[0]['relevance']:.3f} >= threshold {threshold:.3f}"
        return {
            "decision": status,
            "reason": reason,
            "trust": None,
            "guarantee": (answer_check or {}).get("guarantee"),
            "versions": {"ranker": Ranker.model().get("source", ""), "embedding": self.semantic.model_name,
                         "extractor": EXTRACTOR_VERSION, "answer_check": (answer_check or {}).get("model")},
        }

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
