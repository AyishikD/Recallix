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
from ..features import reranker as rerank_cfg
from ..features import aggregation_check
from ..features import answer_verify
from ..features import modes
from ..features.reranker import Reranker
from ..features.implicit import ImplicitInference, rules_config as implicit_config
from ..features.facts import EXTRACTOR_VERSION
from . import freshness
from ..features import trust
from ..features import dates as date_resolution
from .updates import _norm_answer, about_the_user, asks_about_now, ended_values, matches_purpose, promote_updates, purpose_stems, resolve_answer_conflicts, states_a_change, user_perspective, without_other_people

_INTERNAL_KEYS = ("_query_slot", "_slot_id", "_slot_value", "features", "semantic_score", "bm25_score",
                  "symbolic_match", "pool", "_answer_ok", "_counts")


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

    def _after_decision(self, user_id, agent_id, query, final, resolved, seen_ids, q_vec, analysis, top_k, deep=None):
        """
        Refills the places after the first results once recall has decided to answer:
        recall.second_pass proposes memories the first ranking under-ranks (a second search and the
        messages around the best result), then recall.rerank (when `deep`, or on by config) reorders those places by
        reading each memory with the question. The first `keep` results, and any later one the reader
        found the answer in, stay where they are.
        """
        sp = settings.get("recall.second_pass", {}) or {}
        rr = rerank_cfg.config()
        rerank_after = (rr["enabled"] or (rr["auto"] and self._needs_deep(query))) if deep is None else bool(deep)
        keep = rr["keep"] if rerank_after else int(sp.get("keep", 5))
        if top_k <= keep or not (sp.get("enabled", False) or rerank_after):
            return final
        head = final[:keep] + [m for m in final[keep:] if m.get("_answer_ok")]
        kept = {m["id"] for m in head}
        rest = [m for m in final[keep:] if m["id"] not in kept]
        rest += [m for m in resolved if m["id"] not in kept and m["id"] not in {r["id"] for r in rest}]
        ordered = head + rest
        if sp.get("enabled", False) and q_vec is not None:
            ordered = self._second_pass(user_id, agent_id, ordered, len(head), seen_ids, q_vec, analysis)
        if rerank_after:
            ordered = self._rerank(query, ordered, len(head), user_id, agent_id)
        return ordered[:top_k]

    def _rerank(self, query, ordered, n_head, user_id=None, agent_id=None):
        """
        recall.rerank: the `candidates` memories after the first `n_head` are reordered by a
        cross-encoder that reads each together with the question (features/reranker.py); the first
        `n_head` and everything after the re-read ones keep their places. With `context`, each
        memory is also read after the message stored just before it, and the better of the two
        scores counts: a short reply ("yes, last week!") only answers next to what it replies to.
        """
        model = Reranker.from_config(force=True)
        if model is None or len(ordered) <= n_head + 1:
            return ordered
        cfg = rerank_cfg.config()
        n = cfg["candidates"]
        head, pool, tail = ordered[:n_head], ordered[n_head:n_head + n], ordered[n_head + n:]
        texts = [self._with_dates(m) for m in pool]
        scores = model.scores(query, texts)
        if cfg["context"] and user_id is not None:
            before = self.episodic.previous_of(user_id, [m["id"] for m in pool], agent_id, cfg["context_gap_minutes"] * 60.0)
            short = cfg["context_max_words"]  # only short replies need the message before them (None: all)
            idx = [i for i, m in enumerate(pool) if m["id"] in before
                   and (short is None or len((m.get("content") or "").split()) <= short)]
            if idx:
                with_ctx = model.scores(query, [f"{before[pool[i]['id']]}\n{texts[i]}" for i in idx])
                for i, sc in zip(idx, with_ctx):
                    scores[i] = max(scores[i], sc)
        for m, sc in zip(pool, scores):
            m["rerank_score"] = round(sc, 3)
        pool = [m for _, m in sorted(zip(scores, pool), key=lambda t: -t[0])]  # sorted() is stable on ties
        return head + pool + tail

    def _second_pass(self, user_id, agent_id, ordered, n_head, seen_ids, q_vec, analysis):
        """
        recall.second_pass: of the places after the first `n_head`, `search` go to the memories
        closest in meaning to the question and the top results together (the question alone
        misses a second fact that shares few words with it); then come the messages stored right
        around the best-ranked result (the answer is often one line away from the best match);
        then the rest, in the second search's order. Returns the whole list, head unchanged.
        """
        cfg = settings.get("recall.second_pass", {}) or {}
        import numpy as np
        unit = lambda v: v / (float(np.linalg.norm(v)) or 1.0)  # noqa: E731
        head, tail = ordered[:n_head], ordered[n_head:]
        if not head or not tail:
            return ordered
        vecs = self.episodic.get_embeddings([m["id"] for m in ordered], self.semantic.model_name)
        # anchored on the best-ranked results: the reader may have moved an answering memory to the front
        by_rank = sorted(head, key=lambda m: -m["relevance"])
        seeds = [vecs[m["id"]] for m in by_rank[:int(cfg.get("seeds", 3))] if m["id"] in vecs]
        if seeds:
            q2 = unit(unit(np.asarray(q_vec, dtype=np.float32)) + float(cfg.get("weight", 0.5)) * unit(np.mean(seeds, axis=0)))
            sim = {m["id"]: float(np.dot(vecs[m["id"]], q2)) if m["id"] in vecs else -1.0 for m in tail}
            tail = sorted(tail, key=lambda m: -sim[m["id"]])  # stable: ties keep the first ranking
        picked = tail[:int(cfg.get("search", 2))] if seeds else []
        for m in picked:
            m["found_by"] = "second_search"
        taken = {m["id"] for m in head} | {m["id"] for m in picked}
        by_id = {m["id"]: m for m in tail}
        near = []
        window = int(cfg.get("neighbours", 2))
        if window > 0:
            rows = self.episodic.neighbours(user_id, by_rank[0]["id"], agent_id, window,
                                            float(cfg.get("max_gap_minutes", 60)) * 60.0)
            fresh = [r for r in rows if r["id"] not in seen_ids]  # not a candidate of the first search: score it now
            if fresh:
                self._attach_semantic_scores(q_vec, fresh)
                self._score(fresh, analysis, agent_id)
            for r in rows:
                if r["id"] in taken:
                    continue
                m = by_id.get(r["id"]) or (r if r["id"] not in seen_ids else None)  # None: removed by truth arbitration
                if m is not None:
                    m["found_by"], m["neighbour_of"] = "neighbour", by_rank[0]["id"]
                    near.append(m)
                    taken.add(m["id"])
        return head + picked + near + [m for m in tail if m["id"] not in taken]

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
    def multi_stage_recall(self, user_id, query, agent_id="default_agent", top_k=5, return_candidates=False, deep=None,
                           keep_on_decline=None, mode=None):
        """
        mode: "cautious" | "balanced" | "helpful" for this recall (features/modes.py); None = config recall.mode.
        """
        token = modes.use(mode)
        try:
            out = self._recall(user_id, query, agent_id, top_k, return_candidates, deep, keep_on_decline)
        finally:
            modes.reset(token)
        return out

    def _recall(self, user_id, query, agent_id="default_agent", top_k=5, return_candidates=False, deep=None,
                keep_on_decline=None):
        """
        deep: True reads the candidates with the re-ranker (recall.rerank: better on questions whose
        answer is spread over several memories, slower); False never; None follows the config.
        """
        from .reasoning_engine import ReasoningEngine
        from .truth_engine import TruthEngine
        t0 = time.time()

        asked = query
        query = user_perspective(query, self._NOT_PEOPLE, lambda name: self.episodic.has_speaker(user_id, name))
        # "does she work" -> "do I work" (recall.user_perspective)
        effective_query, alias_sources = self._resolve_aliases(user_id, query, agent_id)
        aliases = [effective_query] if effective_query != asked else []
        analysis = analyze_query(effective_query)
        q_vec = self.semantic.embed(effective_query)

        candidates = self._generate_candidates(user_id, effective_query, agent_id, q_vec)
        ranked = promote_updates(self._score(candidates, analysis, agent_id, alias_sources), effective_query)  # recall.updates
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
        # keep_on_decline (per request; None = recall.keep_results_on_decline, off by default): a decline
        # still returns the ranked results (status ABSTAINED, answer None), so a caller that checks the
        # status can show them as low-confidence instead of nothing
        if keep_on_decline is None:
            keep_on_decline = settings.get("recall.keep_results_on_decline", False)
        keep_on_decline = bool(keep_on_decline)
        checker = AnswerChecker.from_config()
        outsider = self._unknown_person(asked, all_scored)
        # recall.aggregation_check: for counting / ordering / time-between questions, whether the memories
        # support every event the question presupposes. mode "decide": that alone answers or declines.
        # mode "rescue" (the default when enabled): it only turns a decline of the usual path into an answer,
        # further down; it never declines anything itself.
        agg_mode = aggregation_check.config()["mode"]
        aggregation = self._aggregation_check(asked, resolved) if resolved and not outsider and agg_mode == "decide" else None
        if outsider:
            status, final = "ABSTAINED", []
        elif not resolved:
            status, final = "NO_MATCH", []
        elif aggregation is not None:
            status = "CONSISTENT" if aggregation["supported"] else "ABSTAINED"
            final = resolved[:top_k] if (aggregation["supported"] or keep_on_decline) else []
        elif checker is not None:
            final = resolved[:top_k]
            t = checker.threshold()
            accept, subject = self._subject_contrast(checker, asked, all_scored)
            conflicts = (settings.get("recall.answer_conflicts", {}) or {})
            min_read = int(conflicts.get("read", 3)) if conflicts.get("enabled", True) else 1
            speakers = list(dict.fromkeys(sp for sp in (memory_speaker(m) for m in all_scored[:200]) if sp))
            named = [sp for sp in speakers if re.search(rf"\b{re.escape(sp)}\b", asked, re.I)]
            person = named[0] if len(named) == 1 and len(speakers) > 1 and named[0].lower() not in self._ROLE_NAMES else None
            read, best, answered = self._read_until_answer(checker, asked, final, t, accept, min_read, speakers, person)  # the reader keeps the question as asked
            cert = checker.certificate() if hasattr(checker, "certificate") else None  # custom checkers may omit it
            answer_check = {"model": checker.model_name, "threshold": t, "best_margin": round(best, 3),
                            "read": len(read), "guarantee": cert["statement"] if cert else None,
                            # best margin among answers the rules accept (-100: none): what the threshold acts on
                            "counted_margin": round(max((m["answer_margin"] for m in read if m.get("_counts")), default=-100.0), 3),
                            "subject": subject,
                            "rejected_as_about_someone_else": [m["id"] for m in read if not m.get("_answer_ok") and
                                                               m.get("subject_contrast") is not None]}
            if answered:
                # recall.answer_check.verify: the memory the reader answered from must actually answer the question
                verifier = answer_verify.AnswerVerifier.from_config()
                if verifier is not None:
                    accepted = [m for m in read if m.get("_answer_ok")]
                    scores = verifier.scores(asked, [answer_verify.premise(m.get("content", ""), memory_speaker(m))
                                                     for m in accepted])
                    vt = answer_verify.config()["threshold"]
                    for m, sc in zip(accepted, scores):
                        m["answer_verified"] = round(sc, 5)
                        if sc < vt:
                            m["_answer_ok"] = False  # the best-answer pick and the receipt skip it
                    answer_check["verify"] = {"model": verifier.model_name, "threshold": vt,
                                              "scores": {m["id"]: round(sc, 5) for m, sc in zip(accepted, scores)}}
                    if accepted and not any(sc >= vt for sc in scores):
                        answered = False
                        answer_check["unverified"] = [m["id"] for m in accepted]
            status = "CONSISTENT" if answered else "ABSTAINED"
            if status == "ABSTAINED":
                inferred = self._answer_by_inference(checker, asked, resolved[:3], t)
                if inferred is not None:
                    status, answer_check["inferred_from"] = "CONSISTENT", inferred["id"]
                    final = [inferred] + [m for m in resolved[:top_k] if m["id"] != inferred["id"]][: top_k - 1]
            if status == "ABSTAINED" and self._needs_calculation(asked) and not answer_check["rejected_as_about_someone_else"] \
                    and resolved[0]["relevance"] >= threshold:
                # recall.answer_check.calculation_questions: "how many times ...", "how long since ..." are
                # answered by counting or subtracting across memories, never by a phrase inside one, so
                # the reader cannot confirm them. The relevance threshold decides instead, and the
                # result says the answer still has to be worked out.
                status, final, answer_check["needs_calculation"] = "CONSISTENT", resolved[:top_k], True
            if status == "ABSTAINED":
                final = resolved[:top_k] if keep_on_decline else []
            elif (settings.get("recall.answer_check", {}) or {}).get("rerank", False):
                # memories in which the reader found an answer first (stable: keeps relevance order within)
                final = sorted(final, key=lambda m: not m.get("_answer_ok"))
            if status == "CONSISTENT" and not answer_check.get("inferred_from"):
                final = resolve_answer_conflicts(final, asked)  # recall.answer_conflicts: later answer first
        elif resolved[0]["relevance"] < threshold:
            status, final = "ABSTAINED", (resolved[:top_k] if keep_on_decline else [])
        else:
            status, final = "CONSISTENT", resolved[:top_k]

        if status == "ABSTAINED" and agg_mode == "rescue" and resolved and not outsider:
            # Held-out LongMemEval: deciding these questions outright refused more real questions than it
            # caught traps, while rescuing the usual path's declines cut wrong refusals 78 -> 55 of 470 with
            # no unanswerable question answered (eval/results/aggregation_check/README.md).
            rescue = self._aggregation_check(asked, resolved)
            if rescue is not None and rescue["supported"]:
                rescue["rescued"] = True
                aggregation, status, final = rescue, "CONSISTENT", resolved[:top_k]
        computed = None
        agg_cfg = aggregation_check.config()
        if status == "CONSISTENT" and resolved and not outsider and agg_cfg["enabled"] and agg_cfg["compute"]:
            # recall.aggregation_check.compute: for time-between and ordering questions, work the answer out from
            # the memories the check verified (exact dates, shown in the receipt); never changes the decision
            basis = aggregation
            if basis is None:
                parsed = aggregation_check.presuppositions(asked)
                if parsed is not None and parsed["kind"] in ("time_between", "ordering"):
                    basis = self._aggregation_check(asked, resolved)
            computed = aggregation_check.compute(basis, {m["id"]: m for m in resolved}, agg_cfg["compute_min_support"])

        if status == "CONSISTENT":
            # recall.second_pass and recall.rerank: applied after the answer-or-decline
            # decision, so they cannot change it
            final = self._after_decision(user_id, agent_id, asked, final, resolved, {r["id"] for r in ranked},
                                         q_vec, analysis, top_k, deep)

        recall_id = uuid.uuid4().hex
        self.recall_log.log(recall_id, user_id, all_scored)
        self.recall_log.log_query(recall_id, user_id, agent_id, asked)
        if answer_check is not None:
            self.recall_log.log_answer_check(recall_id, user_id, answer_check["model"], answer_check["counted_margin"],
                                             [m["id"] for m in read])
        self._reinforce(final if status == "CONSISTENT" else [])  # kept-on-decline results were not used

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
            resolved_text = self._with_dates(out)
            if resolved_text != out.get("content"):
                out["content_resolved"] = resolved_text  # memory.resolve_dates: "yesterday [7 May 2023]"
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
            "answer": self._best_answer(final, memories, asked) if status == "CONSISTENT" else None,
            "memories": memories,
            "receipt": self._recall_receipt(status, resolved, final, threshold, answer_check, outsider, aggregation),
            "debug": {
                "aliases": aliases,
                "read_as": query if query != asked else None,
                "promoted_updates": [{"memory": m["id"], "update_of": m["update_of"], "shared": m.get("update_shared_words")}
                                     for m in all_scored[:5] if m.get("update_of") is not None],
                "query_entities": analysis["strong_entities"],
                "candidates": len(all_scored),
                "threshold": round(threshold, 4),
                "top_relevance": round(resolved[0]["relevance"], 4) if resolved else None,
                "model": Ranker.model().get("source", ""),
                "answer_check": answer_check,
                "aggregation_check": aggregation,
                "mode": modes.current(),
            },
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
        if computed is not None:
            result["computed_answer"] = computed
            result["receipt"]["computed"] = computed["working"]
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

    # Counting occurrences, or the time between / since events. Not "how many employees ..." or
    # "how much rent ...": those are single facts the reader can find (and a trap if never said).
    _CALCULATION = re.compile(
        r"\bhow many times\b"
        r"|\bhow many (?:days|weeks|months|years) (?:passed|have passed|between|since|ago|apart|has it been)\b"
        r"|\bhow long (?:ago|has it been|since|passed|between)\b", re.I)

    # Questions whose answer is usually spread over several memories: lists, kinds, "what has X done",
    # two people, "in common". recall.rerank.auto makes these deep recalls.
    _SPREAD_OUT = re.compile(
        r"\bwhat (?:kinds?|types?|sorts?|activities|things|are some|events|hobbies|sports|books|places|countries|cities|"
        r"pets|items|ways|games|foods|songs|movies|shows|instruments|classes|plans)\b"
        r"|\b(?:what|which) .*\b(?:has|have|did) .* (?:done|tried|pursued|made|visited|attended|participated|bought|read|"
        r"watched|played|taken|learned|joined)\b|\bboth\b|\bin common\b|\b[A-Z][a-z]+ and [A-Z][a-z]+\b", re.I)

    @classmethod
    def _needs_deep(cls, query):
        return bool(cls._SPREAD_OUT.search(str(query or "")))

    @classmethod
    def _needs_calculation(cls, query):
        on = (settings.get("recall.answer_check", {}) or {}).get("calculation_questions", False)
        return bool(on and cls._CALCULATION.search(str(query or "")))

    @staticmethod
    def _aggregation_check(query, resolved):
        """
        recall.aggregation_check (features/aggregation_check.py). None when it is off, the entailment model
        is unavailable, or the question is not a counting / ordering / time-between question; otherwise
        {"kind", "supported", "claims", "model", "threshold", "needs_calculation": True, "read": memories read}.
        """
        cfg = aggregation_check.config()
        if not cfg["enabled"] or aggregation_check.presuppositions(query) is None:
            return None
        checker = aggregation_check.EntailmentChecker.from_config()
        if checker is None:
            return None
        memories = resolved[:cfg["memories"]]
        out = aggregation_check.check(checker, query, memories, cfg["threshold"],
                                      read=lambda m: without_other_people(m.get("content", ""), query),
                                      speaker=memory_speaker)
        if out is not None:
            out.update(model=checker.model_name, threshold=cfg["threshold"], needs_calculation=True, read=len(memories))
        return out

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

    _VOCATIVE_CUE = r"(?:hey|hi|hello|thanks|thank you|congrats|congratulations|wow|yeah|yes|oh|aw+|omg|sorry|good luck)"

    _SECOND_PERSON = re.compile(r"\b(?:you|your|you're|you've|you'll|yours|yourself)\b", re.I)
    _FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|me|mine|myself|we|our|us|we're|we've)\b", re.I)

    @classmethod
    def _about_the_listener(cls, memory, subject):
        """
        The asked person's own message that speaks only to the listener ("That charity race sounds
        great, Mel! Making a difference for mental health is huge.") is about the listener, not
        about them. Checked on the stored text (before "I" is rewritten to the speaker's name).
        """
        if not subject or memory_speaker(memory) != subject:  # (role names never reach here: see the caller)
            return False
        body = re.sub(r"^\s*(?:\[[^\]]*\]\s*)?[^:\n]{1,40}:\s*", "", memory.get("content", ""))
        return bool(cls._SECOND_PERSON.search(body)) and not cls._FIRST_PERSON.search(body)

    _ROLE_NAMES = {"user", "assistant", "ai", "bot", "system", "agent", "human", "me", "you"}

    @classmethod
    def _answer_says_nothing(cls, answer, context, speakers, subject=None):
        """
        recall.answer_check.reject_non_statements: spans a SQuAD reader takes from text that states no
        fact, measured on LoCoMo dev to be (almost) never the right answer:
          * the answer lies only inside questions ("Wow, that looks amazing! What inspired it?"): the
            speaker is asking, so the fact (if any) belongs to someone else;
          * the answer is a speaker's name used to address them ("Thanks, Melanie!", "Hey Caroline,").
        (_about_the_listener is the third case.)
        """
        a = (answer or "").strip(" .,!?;:'\"")
        if not a:
            return False
        body = re.sub(r"^\s*(?:\[[^\]]*\]\s*)?[^:\n]{1,40}:\s*", "", context or "")
        hits = [x for x in re.split(r"(?<=[.!?])\s+", body) if a.lower() in x.lower()]
        # Only for questions about a named participant of a conversation between people: an assistant
        # often suggests things as questions ("Have you tried Kyoto?"), and then that is the answer.
        if subject and hits and all(x.rstrip().endswith("?") for x in hits):
            return True
        if any(a.lower() == sp.lower() for sp in speakers):
            name = re.escape(a)
            if re.search(rf"\b{cls._VOCATIVE_CUE}\b[\s,]*{name}\b|\b{name}\s*[!?]|,\s*{name}\s*[,.!?]|^\s*{name}\s*,", body, re.I):
                return True
        return False

    @staticmethod
    def _best_answer(final, memories, query=""):
        """
        The answer to show: the span the reader was most confident in among the memories it accepted.
        Measured on LoCoMo dev (eval/results/retrieval_fixes): answer token-F1 0.339 -> 0.375 and
        time questions answered with the gold date 59 -> 68 of 156 over preferring the
        recall.answer_conflicts pick; our dev set and the Gemini batches unchanged. When that check
        found a later, different answer (a possible update without a cue word), it is reported as
        `later_answer` (and its memory stays first in `memories`); best_answer: "conflict" restores
        the old choice. {"text", "memory_id", "margin"[, "later_answer", "later_memory_id"]} or None.

        A memory that a later message explicitly updates (promote_updates marks the later one with
        `update_of`: "update: ballet got rescheduled to wednesdays" updates "ballet on tuesdays") is not
        a candidate: its span is the outdated fact, however confident the reader is in it. If only
        outdated memories hold an answer (the update says "cancelled", with no new value), there is no
        top-level answer rather than the outdated one.
        """
        accepted = [(m, out) for m, out in zip(final, memories) if m.get("_answer_ok") and m.get("answer")]
        outdated = {m.get("update_of") for m in final if m.get("update_of") is not None}
        accepted = [(m, out) for m, out in accepted if m.get("id", out.get("id")) not in outdated]
        if not accepted:
            return None
        if asks_about_now(query):
            # a value a later message says has ended ("resigned from infosys", "sold the honda") is not the user's
            # present: no answer rather than the outdated one
            ended = ended_values(final)
            def still_current(m):
                words = {w for w in re.findall(r"[a-z0-9]+", str(m.get("answer") or "").lower()) if len(w) >= 3}
                ts = float(m.get("timestamp") or 0)
                return not any(t >= ts and words & e for t, e in ended)
            accepted = [(m, out) for m, out in accepted if still_current(m)]
            if not accepted:
                return None
            # "Where do I live?": when memories give different answers, the user's latest statement is their present,
            # even when the two share no words to be paired as an update ("i live in delhi" ... "found a flat in pune
            # near my new office"). eval/results/update_set
            # Only answers stated about the user, and confident enough to replace an earlier one, can override it
            # (recall.answer_conflicts.override_min_margin; acceptance alone is a much lower bar): "I moved to Mumbai"
            # does not answer "Where do I work?" over "I work at Zomato", nor "Anita lives in Bangalore" "Where do I live?".
            answers = {_norm_answer(m.get("answer")) for m, _ in accepted}
            floor = float((settings.get("recall.answer_conflicts", {}) or {}).get("override_min_margin", -0.5))
            confident = max(accepted, key=lambda p: p[0].get("answer_margin", float("-inf")))
            pool = [p for p in accepted if p is confident or
                    (about_the_user(p[0].get("content")) and float(p[0].get("answer_margin") or -99) >= floor)]
            # "Which credit card do I use for booking flights?": when the question narrows the topic and some memories
            # match it ("hdfc regalia for all flight bookings"), a later one that does not ("sbi cashback card for
            # swiggy orders") is a different fact, not an update. eval/results/update_set_fresh
            stems = purpose_stems(query)
            if any(matches_purpose(p[0].get("content"), stems) for p in pool):
                pool = [p for p in pool if matches_purpose(p[0].get("content"), stems)]
            if len(answers) > 1 and len(pool) > 1:
                m, out = max(pool, key=lambda p: float(p[0].get("timestamp") or 0))
                return {"text": m["answer"], "memory_id": out.get("id"), "margin": m.get("answer_margin")}
        conflict = next((p for p in accepted if p[0].get("answer_conflict_with")), None)
        policy = (settings.get("recall.answer_conflicts", {}) or {}).get("best_answer", "confidence")
        if conflict and policy == "confidence" and asks_about_now(query):
            policy = "conflict"  # "who is MY manager?": the later answer is the current fact (see asks_about_now)
        if conflict and policy == "confidence":
            by_id = {out.get("id"): m for m, out in zip(final, memories)}
            earlier = [by_id.get(i, {}).get("content") for i in conflict[0].get("answer_conflict_with") or []]
            if any(states_a_change(conflict[0].get("content"), e, query) for e in earlier if e):
                policy = "conflict"  # "shifted meera to inventure academy": a stated change for the person asked about
        m, out = conflict if conflict and policy == "conflict" else max(
            accepted, key=lambda p: p[0].get("answer_margin", float("-inf")))
        best = {"text": m["answer"], "memory_id": out.get("id"), "margin": m.get("answer_margin")}
        if conflict and conflict[1].get("id") != best["memory_id"]:
            best["later_answer"], best["later_memory_id"] = conflict[0]["answer"], conflict[1].get("id")
        return best

    @staticmethod
    def _with_resolved_date(answer, context):
        """
        memory.resolve_dates writes "last weekend [the weekend before 17 July 2023]". A reader span
        that ends right before such a bracket ("last weekend") is answered with the resolved date.
        """
        if not answer or not context:
            return answer
        i = context.find(answer)
        if i < 0:
            return answer
        m = re.match(r"\s*\[([^\[\]]{3,80})\]", context[i + len(answer):])
        return m.group(1) if m else answer

    @staticmethod
    def _read_until_answer(checker, query, memories, threshold, accept=None, min_read=1, speakers=(), subject=None):
        """
        Reads memories in rank order — 1, then 2 at a time — and stops at the first batch in
        which one contains an answer (margin >= threshold). The decision ("does any served
        memory answer?") is identical to reading all of them, but an answered recall usually
        needs one read instead of top_k. Declines still read everything. With a non-finite or
        calibration-mode threshold (<= -1e8) everything is read, so margins stay complete.
        `accept(memory, context)`, if given, must also hold for a memory to count as answering
        (the subject-contrast check). At least `min_read` memories are read even when the first
        answers, so disagreeing answers can be found (recall.answer_conflicts).
        Returns (memories read, best margin, answered).
        """
        # the reader sees first-person statements resolved to their recorded speaker, and without
        # clauses about other people the question does not ask about (recall.answer_check.other_people)
        contexts = [without_other_people(resolve_first_person(RecallEngine._with_dates(m), memory_speaker(m)), query)
                    for m in memories]
        early_exit = threshold > -1e8
        read, best, answered, i = [], float("-inf"), False, 0
        while i < len(memories):
            size = (max(1, min_read) if i == 0 else 2) if early_exit else len(memories) - i
            batch, ctxs = memories[i:i + size], contexts[i:i + size]
            results = checker.check(query, [c or "." for c in ctxs])
            for m, c, ctx in zip(batch, results, ctxs):
                if not ctx:  # entirely about someone else the question does not ask about
                    c = {"margin": -100.0, "answer": ""}
                    m["about_someone_else"] = True
                m["answer"], m["answer_margin"] = RecallEngine._with_resolved_date(c["answer"], ctx), round(c["margin"], 3)
                best = max(best, c["margin"])
            reject = (settings.get("recall.answer_check", {}) or {}).get("reject_non_statements", False)
            for m, ctx in zip(batch, ctxs):
                # `_counts`: the rules would accept this memory's answer, whatever the threshold. The cheap
                # rules are checked for every memory; the subject-contrast check (another reader pass) only
                # where the threshold is met, as before. Its margin is what the threshold acts on
                # (counted_margin, logged for learning the threshold: answer_check_learning.py).
                if reject and (RecallEngine._answer_says_nothing(m["answer"], ctx, speakers, subject)
                               or RecallEngine._about_the_listener(m, subject)):
                    if m["answer_margin"] >= threshold:
                        m["answer_not_a_statement"] = True
                    continue
                m["_counts"] = True
                if m["answer_margin"] >= threshold:
                    if accept is not None and not accept(m, ctx):
                        m["_counts"] = False
                        continue
                    m["_answer_ok"] = True
                    answered = True
            read.extend(batch)
            i += size
            if early_exit and answered:
                break
        return read, best, answered

    @staticmethod
    def _with_dates(m):
        """The memory's text with its resolved relative dates inline (memory.resolve_dates), else its text."""
        meta = m.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {}
        found = (meta or {}).get("dates") if isinstance(meta, dict) else None
        return date_resolution.annotate(m.get("content", ""), found) if found else m.get("content", "")

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
                "confirmed_for": m.get("confirmed_for"),
                "inferred": m.get("inferred"),
                "answer": m.get("answer") or None, "answer_margin": m.get("answer_margin")}

    def _recall_receipt(self, status, resolved, final, threshold, answer_check, outsider=None, aggregation=None):
        """Why this recall answered or declined, and which models and rules were involved."""
        if outsider:
            reason = f"the question asks about {outsider}, who is not mentioned in any memory"
        elif status == "NO_MATCH":
            reason = "no stored memory matched the question"
        elif aggregation is not None:
            t = aggregation["threshold"]
            missing = [c for c in aggregation["claims"] if c["support"] < t]
            if aggregation.get("rescued"):
                reason = ("the answer check found no answer phrase, but the memories support every event this "
                          f"{aggregation['kind'].replace('_', '-')} question takes for granted: "
                          + "; ".join(f"\"{c['statement']}\" (memory {c['memory']}, support {c['support']})"
                                      for c in aggregation["claims"])
                          + ". The answer has to be worked out across the memories")
            elif missing:
                reason = (f"this {aggregation['kind'].replace('_', '-')} question takes for granted "
                          + "; ".join(f"\"{c['statement']}\" (support {c['support']})" for c in missing)
                          + f", which no memory supports (threshold {t})"
                          + "".join(f". Memory {c['plan_only']['memory']} only records a plan for it, not the event"
                                    for c in missing if c.get("plan_only")))
            else:
                reason = (f"the memories support every event this {aggregation['kind'].replace('_', '-')} question takes "
                          "for granted: " + "; ".join(f"\"{c['statement']}\" (memory {c['memory']}, support {c['support']})"
                                                      for c in aggregation["claims"])
                          + ". The answer has to be worked out across the memories")
        elif answer_check is not None:
            t, b = answer_check["threshold"], answer_check["best_margin"]
            rejected = answer_check.get("rejected_as_about_someone_else") or []
            src = next((m for m in final if m.get("inferred")), None) if answer_check.get("inferred_from") else None
            if src is not None:
                inf = src["inferred"]
                reason = (f"answer inferred from memory {src['id']}: \"{src.get('content', '')[:80]}\" implies "
                          f"\"{inf['statement']}\" (commonsense pattern \"{inf['pattern']}\" -> {inf['relation']}; "
                          f"entailment {inf['entailment']})")
            elif status == "ABSTAINED" and answer_check.get("unverified"):
                v = answer_check["verify"]
                reason = (f"the reader found answer-like spans in memories {answer_check['unverified']}, but none of them "
                          f"answers the question (verifier {v['model']}: best score "
                          f"{max(v['scores'].values()):.5f} < {v['threshold']})")
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
            "mode": modes.current(),
            "reason": reason,
            "trust": None,
            "guarantee": (answer_check or {}).get("guarantee"),
            "versions": {"ranker": Ranker.model().get("source", ""), "embedding": self.semantic.model_name,
                         "extractor": EXTRACTOR_VERSION, "answer_check": (answer_check or {}).get("model"),
                         "aggregation_check": (aggregation or {}).get("model")},
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
