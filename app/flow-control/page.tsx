"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
    Brain, Cpu, Database, ArrowLeft, Search, RefreshCw, Zap, Users, Shield, TrendingUp, Eye,
    Archive, Sparkles, Target, GitBranch, Filter, Scale, Copy,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { headline, ms, pct, results } from "@/lib/results";

/* Every description below matches the code; every number comes from eval/results/summary.json. */

type System = "python" | "cpp" | "sqlite" | "embedding";
interface Stage {
    title: string;
    subtitle: string;
    detail: string;
    code: string;
    icon: React.ReactNode;
    system: System;
    logStage?: string; // stage name in the store processing_log, for the measured duration
}

const STORE: Stage[] = [
    { title: "Authenticate", subtitle: "Key → user", system: "python", icon: <Shield className="w-5 h-5" />,
      detail: "The X-API-Key is hashed (SHA-256) and looked up. A user key acts only as itself; service/admin keys may pass user_id for an end user. Writes are rate-limited per user.",
      code: `principal = keys.lookup(request.headers["X-API-Key"])\nuser_id = acting_user(principal, body.user_id)  # 403 if not allowed\nrate_limit_writes(user_id)                       # 429 above the limit` },
    { title: "Deduplication", subtitle: "Persistent, agent-aware", system: "sqlite", icon: <Copy className="w-5 h-5" />, logStage: "Deduplication",
      detail: "Identical content already visible to this agent is not stored again (SHA-256 content hash, indexed per user).",
      code: `SELECT id FROM episodic_events\nWHERE user_id = ? AND content_hash = ?\n  AND (memory_type = 'shared' OR agent_id = ?)` },
    { title: "Sensory & working memory", subtitle: "Recent context", system: "python", icon: <Eye className="w-5 h-5" />, logStage: "Sensory & Working Memory",
      detail: "The message enters a per-user ring buffer (TTL from config) and a sliding window of recent messages stored as one SQLite row per user.",
      code: `sensory.add(user_id, message)\nworking.update(user_id, message)   # window = memory.working_window` },
    { title: "Embedding", subtitle: "Computed once", system: "embedding", icon: <Cpu className="w-5 h-5" />, logStage: "Embedding",
      detail: "One embedding per message from the configured provider (Ollama, in-process sentence-transformers, or an API). It is reused by every later stage and stored with the model name.",
      code: `vector = semantic.embed(message)   # unit-normalised, LRU-cached` },
    { title: "Fact extraction", subtitle: "Rule grammar", system: "python", icon: <GitBranch className="w-5 h-5" />, logStage: "Fact Extraction",
      detail: "Patterns such as 'my X is Y', 'X's Y is Z', 'the Y of X is Z', 'X deadline moved to Z' produce (entity, relation, value) slots. A precision guard rejects casual statements ('my cat is sleeping').",
      code: `slots = to_slots(extract_triads(message))\n# "I live in Pune" -> user::location = Pune` },
    { title: "Schema", subtitle: "Prototype classifier", system: "python", icon: <Target className="w-5 h-5" />, logStage: "Schema Classification",
      detail: "Cosine similarity to seed-sentence prototypes of each schema (identity, preference, calendar, learning, task, security, social, general), plus small lexical cue bonuses.",
      code: `score_k = 0.5·cos(v, centroid_k) + 0.5·max_j cos(v, seed_kj) + 0.04·cues_k\nlabel = argmax_k score_k` },
    { title: "Importance", subtitle: "Learned logistic model", system: "python", icon: <TrendingUp className="w-5 h-5" />, logStage: "Importance Scoring",
      detail: "Novelty (distance to the user's nearest memory), information content (IDF), specificity and cue words, combined by weights fitted on labelled data.",
      code: `importance = 1 + 9·sigmoid(w · [novelty, idf, specificity, goal, preference, explicit, small_talk, brevity])` },
    { title: "Episodic storage", subtitle: "System of record", system: "sqlite", icon: <Database className="w-5 h-5" />, logStage: "Episodic Storage",
      detail: "Row + embedding + FTS5 entry in SQLite (WAL), plus a timeline event. SQLite is the source of truth; the vector index is derived from it.",
      code: `INSERT INTO episodic_events (..., embedding, embedding_model, content_hash)\nINSERT INTO episodic_fts (content_id, content, user_id)` },
    { title: "Vector index", subtitle: "Per-user HNSW", system: "cpp", icon: <Zap className="w-5 h-5" />, logStage: "Vector Engine",
      detail: "The vector is appended to the user's namespace in the C++ engine. Pending vectors are scanned exactly, so the write is searchable immediately; a background thread links it into the graph.",
      code: `POST /add {"namespace": user_id, "id": memory_id, "vector": [...]}` },
    { title: "Long-term promotion", subtitle: "High importance", system: "python", icon: <Archive className="w-5 h-5" />, logStage: "Long-Term Promotion",
      detail: "Memories at or above memory.long_term_min_importance are copied to long-term facts with Ebbinghaus decay; deleting the memory deletes the copy.",
      code: `if importance >= cfg.long_term_min_importance:\n    long_term.promote(user_id, event)` },
];

const RECALL: Stage[] = [
    { title: "Alias expansion", subtitle: "From stored facts", system: "python", icon: <Users className="w-5 h-5" />,
      detail: "If a stored memory says 'X is also known as Y', a question mentioning either name is expanded with the other.",
      code: `aliases = slots where relation == "alias_of" and either side occurs in the query` },
    { title: "Candidates", subtitle: "Lexical + vector", system: "sqlite", icon: <Search className="w-5 h-5" />,
      detail: "FTS5 BM25 matches and entity-index hits (SQL, filtered to memories visible to the agent) plus the user's HNSW namespace; vector hits are fetched in one batched query.",
      code: `lexical = episodic.search(user, q, agent, limit=pool)\nvector  = engine /search {namespace: user, top_k: pool}\nrows    = episodic.get_many(user, ids, agent)` },
    { title: "Relevance model", subtitle: "Learned, explainable", system: "python", icon: <Scale className="w-5 h-5" />,
      detail: "Each candidate gets interpretable features (semantic, lexical, entity match/conflict, relation match/conflict, slot match, salience, recency, alias bridge). A logistic model fitted with leave-one-scenario-out CV turns them into a probability.",
      code: `relevance = sigmoid(w · features)   # returned per memory as "explain"` },
    { title: "Truth arbitration", subtitle: "One value per fact", system: "python", icon: <Filter className="w-5 h-5" />,
      detail: "Memories stating the same fact slot are walked in time order: newer values supersede older ones; 'X is not V' invalidates V only.",
      code: `deadline is Dec 12  →  moved to Dec 18  →  "is NOT Dec 12"\nresult: Dec 18 (verified)` },
    { title: "Abstention", subtitle: "Learned threshold", system: "python", icon: <Brain className="w-5 h-5" />,
      detail: "If the best candidate's relevance is below the threshold chosen on dev data, Recallix returns nothing (status ABSTAINED) instead of a guess.",
      code: `if top.relevance < threshold: return {"status": "ABSTAINED", "memories": []}` },
];

const AGENTS: Stage[] = [
    { title: "Private memory", subtitle: "Owner agent only", system: "sqlite", icon: <Shield className="w-5 h-5" />,
      detail: "memory_type='private' is visible only to the agent that wrote it. Enforced in SQL on every read, not by a score penalty.",
      code: `WHERE e.user_id = ? AND (e.memory_type = 'shared' OR e.agent_id = ?)` },
    { title: "Shared memory", subtitle: "All of the user's agents", system: "sqlite", icon: <Users className="w-5 h-5" />,
      detail: "memory_type='shared' is visible to every agent of the same user. Users never see each other's memories: each user also has a separate vector index.",
      code: `POST /agents/store {"agent_id": "planner", "memory_type": "shared", ...}` },
];

const LEARNING: Stage[] = [
    { title: "Recall log", subtitle: "Features per candidate", system: "sqlite", icon: <Database className="w-5 h-5" />,
      detail: "Each recall stores the feature vectors of its top candidates under a recall_id.",
      code: `recall_log.log(recall_id, user_id, ranked[:20])` },
    { title: "Feedback", subtitle: "Which results helped", system: "python", icon: <Sparkles className="w-5 h-5" />,
      detail: "POST /memory/feedback marks the useful memories of a recall; the other logged candidates become negatives.",
      code: `POST /memory/feedback {"recall_id": "...", "relevant_memory_ids": [42]}` },
    { title: "Refit", subtitle: "MAP logistic regression", system: "python", icon: <RefreshCw className="w-5 h-5" />,
      detail: "fit_ranker refits weights (L2 prior around documented hand-set values) and the abstention threshold; fit_importance learns importance weights from which memories were ever recalled. Fitted models load without a restart.",
      code: `python -m backend.brain.memory_ranking.fit_ranker --from-db\npython -m backend.brain.memory_ranking.fit_importance --from-db` },
];

const JOBS = [
    { name: "State inference", key: "workers.state_inference_minutes", desc: "Latest user::<relation> facts and schema shares become the world state." },
    { name: "Prediction", key: "workers.prediction_minutes", desc: "First-order Markov chain over the user's schema sequence (Laplace-smoothed)." },
    { name: "Replay", key: "workers.replay_hours", desc: "Extractive reflection (MMR over embeddings) from long-term facts, with theme shares." },
    { name: "Forgetting", key: "workers.forgetting_hours", desc: "Deletes/archives old, never-recalled, low-importance memories (thresholds in config); removes vectors too." },
];

const systemStyle: Record<System, string> = {
    python: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
    cpp: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    sqlite: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    embedding: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
};

function StageCard({ stage, index, measured }: { stage: Stage; index: number; measured?: number }) {
    const [open, setOpen] = useState(false);
    return (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.04 }}
            onClick={() => setOpen(!open)}
            className="cursor-pointer rounded-2xl border border-white/5 bg-zinc-900/40 p-5 hover:border-white/10 transition-colors">
            <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-zinc-300">{stage.icon}</div>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] font-black text-zinc-600">{String(index + 1).padStart(2, "0")}</span>
                        <h3 className="font-bold text-white">{stage.title}</h3>
                        <span className="text-xs text-zinc-500">{stage.subtitle}</span>
                    </div>
                </div>
                {measured !== undefined && <span className="text-[10px] font-mono text-zinc-400">{ms(measured, 2)} avg</span>}
                <Badge className={`text-[9px] border ${systemStyle[stage.system]}`}>{stage.system}</Badge>
            </div>
            {open && (
                <div className="mt-4 space-y-3">
                    <p className="text-sm text-zinc-400 leading-relaxed">{stage.detail}</p>
                    <pre className="text-xs bg-black/60 border border-zinc-800/50 rounded-xl p-3 text-cyan-300 whitespace-pre-wrap">{stage.code}</pre>
                </div>
            )}
        </motion.div>
    );
}

export default function FlowControlPage() {
    const [tab, setTab] = useState<"store" | "recall" | "agents" | "learning" | "jobs">("store");
    const stageMs: Record<string, number> = headline?.retrieval?.store_stage_ms_mean || {};
    const example = headline?.example;
    const stages = tab === "store" ? STORE : tab === "recall" ? RECALL : tab === "agents" ? AGENTS : LEARNING;
    const tabs = [
        { key: "store" as const, label: "Store", count: `${STORE.length} stages` },
        { key: "recall" as const, label: "Recall", count: `${RECALL.length} stages` },
        { key: "agents" as const, label: "Agents", count: `${AGENTS.length} rules` },
        { key: "learning" as const, label: "Learning loop", count: `${LEARNING.length} steps` },
        { key: "jobs" as const, label: "Background jobs", count: `${JOBS.length} jobs` },
    ];

    return (
        <div className="min-h-screen bg-black text-white">
            <nav className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-white/5 flex items-center px-6 glass backdrop-blur-2xl gap-4">
                <Link href="/" className="flex items-center gap-2 text-zinc-400 hover:text-white"><ArrowLeft className="w-4 h-4" /></Link>
                <span className="text-sm font-black tracking-tight">HOW RECALLIX WORKS</span>
                <span className="ml-auto text-[10px] text-zinc-500 font-bold uppercase tracking-widest">No LLM in the memory pipeline</span>
            </nav>

            <main className="pt-24 pb-24 container mx-auto px-6 max-w-4xl">
                <div className="flex flex-wrap gap-2 mb-8">
                    {tabs.map(t => (
                        <button key={t.key} onClick={() => setTab(t.key)}
                            className={`px-4 py-2 rounded-full text-xs font-bold border transition-colors ${tab === t.key ? "bg-white text-black border-white" : "border-white/10 text-zinc-400 hover:text-white"}`}>
                            {t.label} <span className="opacity-60">· {t.count}</span>
                        </button>
                    ))}
                </div>

                {tab !== "jobs" && (
                    <div className="space-y-3">
                        {stages.map((s, i) => (
                            <StageCard key={s.title} stage={s} index={i} measured={tab === "store" && s.logStage ? stageMs[s.logStage] : undefined} />
                        ))}
                    </div>
                )}

                {tab === "jobs" && (
                    <div className="grid md:grid-cols-2 gap-4">
                        {JOBS.map(j => (
                            <div key={j.name} className="rounded-2xl border border-white/5 bg-zinc-900/40 p-5">
                                <h3 className="font-bold mb-1">{j.name}</h3>
                                <p className="text-sm text-zinc-400 mb-2">{j.desc}</p>
                                <p className="text-[10px] font-mono text-zinc-600">interval: config.yaml {j.key}</p>
                            </div>
                        ))}
                    </div>
                )}

                {tab === "store" && example && (
                    <section className="mt-10 rounded-2xl border border-white/5 bg-zinc-900/40 p-6">
                        <h3 className="font-bold mb-1">Real run</h3>
                        <p className="text-xs text-zinc-500 mb-4">Output of <code>eval/run_eval.py</code> ({headline?.embedder}, {headline?.engine} engine). Stage timings above are averages over the eval stores.</p>
                        <div className="space-y-2">
                            {example.stores.map((s) => (
                                <div key={s.content} className="text-xs font-mono bg-black/50 rounded-lg p-3">
                                    <div className="text-zinc-300">{s.content}</div>
                                    <div className="text-zinc-500 mt-1">importance {s.importance} · schemas {s.schemas.join(", ")} · {s.promoted ? "promoted" : "not promoted"}{s.slots.length ? ` · ${s.slots.join("; ")}` : ""}</div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {tab === "recall" && example && (
                    <section className="mt-10 rounded-2xl border border-white/5 bg-zinc-900/40 p-6">
                        <h3 className="font-bold mb-1">Real run: “{example.query}”</h3>
                        <p className="text-xs text-zinc-500 mb-4">
                            status {example.recall.status} · threshold {example.recall.threshold} · held-out test hit@1 {pct(headline?.retrieval?.["hit@1"])}, recall p50 {ms(headline?.retrieval?.recall_ms_p50, 1)} (incl. embedding)
                        </p>
                        {example.recall.memories.map((m) => (
                            <div key={m.content} className="text-xs font-mono bg-black/50 rounded-lg p-3 mb-2">
                                <div className="text-zinc-300">{m.content} <span className="text-cyan-400">· relevance {m.relevance}</span></div>
                                <div className="text-zinc-500 mt-1">{Object.entries(m.explain).filter(([, v]) => v > 0).map(([k, v]) => `${k}=${v}`).join("  ")}</div>
                            </div>
                        ))}
                    </section>
                )}

                <p className="mt-10 text-[10px] text-zinc-600">Results generated {results.generated_at} at commit {results.git_commit}.</p>
            </main>
        </div>
    );
}
