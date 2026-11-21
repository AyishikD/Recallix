"use client";

import { useState } from "react";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { Eyebrow, Headline, Lede, Reveal, Section } from "@/components/site/ui";
import { headline, ms, results } from "@/lib/results";

/* Every description matches the code; every number comes from eval/results/summary.json. */

interface Stage { title: string; detail: string; code: string; where: string; logStage?: string }

const STORE: Stage[] = [
    { title: "Authenticate", where: "API", detail: "The API key is hashed and looked up. A user key acts only as itself; service and admin keys may act for an end user. Writes are rate-limited per user.",
      code: `principal = keys.lookup(request.headers["X-API-Key"])\nuser_id = acting_user(principal, body.user_id)\nrate_limit_writes(user_id)` },
    { title: "Skip duplicates", where: "SQLite", logStage: "Deduplication", detail: "Identical content already visible to this agent is not stored again.",
      code: `SELECT id FROM episodic_events\nWHERE user_id = ? AND content_hash = ?\n  AND (memory_type = 'shared' OR agent_id = ?)` },
    { title: "Embed once", where: "Embedding model", logStage: "Embedding", detail: "One embedding per message, stored with the model's name and reused by every later step.",
      code: `vector = semantic.embed(message)` },
    { title: "Extract facts", where: "Python", logStage: "Fact Extraction", detail: "A rule grammar turns statements into (entity, relation, value) facts. A precision guard skips casual remarks such as “my cat is sleeping”.",
      code: `"I live in Pune"                  -> user::location = Pune\n"Daedalus reactor core is CRITICAL" -> starshipdaedalus::reactor_core = CRITICAL` },
    { title: "Tag the topic", where: "Python", logStage: "Schema Classification", detail: "The embedding is compared with example sentences for each topic (identity, preference, calendar, …).",
      code: `score_k = 0.5·cos(v, centroid_k) + 0.5·max cos(v, example_kj) + cue bonus` },
    { title: "Score importance", where: "Python", logStage: "Importance Scoring", detail: "Novelty, information content, specificity and cue words, combined by weights fitted on labelled data.",
      code: `importance = 1 + 9·sigmoid(w · features)` },
    { title: "Store", where: "SQLite", logStage: "Episodic Storage", detail: "Row, embedding and full-text index in SQLite — the system of record — plus a timeline entry.",
      code: `INSERT INTO episodic_events (...); INSERT INTO episodic_fts (...)` },
    { title: "Index", where: "C++ engine", logStage: "Vector Engine", detail: "The vector joins the user's own HNSW index. It is searchable immediately and linked into the graph in the background.",
      code: `POST /add {"namespace": user_id, "id": memory_id, "vector": [...]}` },
];

const RECALL: Stage[] = [
    { title: "Resolve aliases", where: "SQLite", detail: "If a memory says “X is also known as Y”, a question using the alias is rewritten to the canonical name.",
      code: `"What powers 'The Dragon's Breath'?" -> "What powers Galaxy X-77?"` },
    { title: "Find candidates", where: "SQLite + C++", detail: "Full-text and entity matches, plus the user's vector index, filtered to memories this agent may see.",
      code: `lexical = fts5 + entity index      # agent-visible, in SQL\nvector  = engine /search {namespace: user}` },
    { title: "Rank", where: "Python", detail: "A logistic model over explainable signals — meaning, keywords, entity and relation matches, who a statement is about, recency — fitted by cross-validation.",
      code: `relevance = sigmoid(w · features)   # returned per memory as "explain"` },
    { title: "Keep facts current", where: "Python", detail: "Memories stating the same fact are walked in time order: newer values win, and “X is not V” removes V only.",
      code: `Dec 12  ->  moved to Dec 18  ->  "not Dec 12"   =>   Dec 18` },
    { title: "Check the answer", where: "QA model", detail: "A small reading-comprehension model (SQuAD 2.0) reads the memories about to be returned, with “I” resolved to whoever said it. Memories that contain an answer come first; if none does, Recallix declines.",
      code: `margin = best answer span − “no answer”\nmargin < threshold for every memory  =>  status ABSTAINED` },
];

const LEARNING: Stage[] = [
    { title: "Log", where: "SQLite", detail: "Each recall records the signals of its top candidates.", code: `recall_log.log(recall_id, user_id, ranked[:20])` },
    { title: "Feedback", where: "API", detail: "Tell Recallix which results helped; the rest become negatives.", code: `POST /memory/feedback {"recall_id": "...", "relevant_memory_ids": [42]}` },
    { title: "Refit", where: "Python", detail: "Ranking and importance weights are refit from your own usage, starting from documented priors. New models load without a restart.",
      code: `python -m backend.brain.memory_ranking.fit_ranker --from-db` },
];

const JOBS = [
    { name: "World state", key: "workers.state_inference_minutes", desc: "Latest facts about the user and the share of each topic." },
    { name: "Prediction", key: "workers.prediction_minutes", desc: "Which topic is likely next, from the user's history." },
    { name: "Reflection", key: "workers.replay_hours", desc: "A short summary of the most representative long-term facts." },
    { name: "Forgetting", key: "workers.forgetting_hours", desc: "Archives or removes old, never-recalled, low-importance memories — vectors included." },
];

function StageList({ stages, timings }: { stages: Stage[]; timings?: Record<string, number> }) {
    const [open, setOpen] = useState<number | null>(0);
    return (
        <div className="rounded-[28px] bg-white shadow-[0_2px_12px_rgba(0,0,0,0.04)] divide-y divide-line/60 overflow-hidden">
            {stages.map((s, i) => (
                <div key={s.title}>
                    <button onClick={() => setOpen(open === i ? null : i)} className="w-full flex items-center gap-5 px-6 md:px-8 py-5 text-left hover:bg-cloud/60 transition-colors">
                        <span className="w-6 text-[13px] text-ink-3 tabular-nums">{String(i + 1).padStart(2, "0")}</span>
                        <span className="flex-1 text-[19px] font-semibold">{s.title}</span>
                        {timings && s.logStage && timings[s.logStage] !== undefined && (
                            <span className="hidden sm:inline text-[13px] text-ink-2 tabular-nums">{ms(timings[s.logStage], 2)}</span>
                        )}
                        <span className="text-[13px] text-ink-3 w-28 text-right">{s.where}</span>
                        <span className={`text-ink-3 transition-transform ${open === i ? "rotate-45" : ""}`} aria-hidden="true">+</span>
                    </button>
                    {open === i && (
                        <div className="px-6 md:px-8 pb-6 pl-[4.25rem] md:pl-[4.75rem]">
                            <p className="text-ink-2 max-w-[640px] mb-4">{s.detail}</p>
                            <pre className="text-[13px] bg-cloud rounded-[14px] p-4 overflow-x-auto whitespace-pre-wrap">{s.code}</pre>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}

export default function HowItWorksPage() {
    const timings = headline?.retrieval?.store_stage_ms_mean;
    const ex = headline?.example;
    return (
        <main>
            <Nav />
            <Section className="text-center">
                <Reveal>
                    <Eyebrow>How it works</Eyebrow>
                    <Headline as="h1">No LLM.<br />Every step explained.</Headline>
                    <Lede className="mx-auto mt-6">
                        Storing and recalling are made of small, inspectable steps. Here is each one — what it does, and the code that does it.
                    </Lede>
                </Reveal>
            </Section>

            <Section tone="cloud">
                <Reveal>
                    <Eyebrow>Store</Eyebrow>
                    <Headline className="mb-10">From message to memory.</Headline>
                </Reveal>
                <StageList stages={STORE} timings={timings} />
                {timings && <p className="mt-4 text-[13px] text-ink-3">Times are averages measured over the evaluation run, including the embedding model.</p>}
                {ex && (
                    <div className="mt-10 grid gap-3 md:grid-cols-2">
                        {ex.stores.map(s => (
                            <div key={s.content} className="rounded-[20px] bg-white p-5 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
                                <p className="font-medium">{s.content}</p>
                                <p className="mt-2 text-[13px] text-ink-2">
                                    importance {s.importance} · {s.schemas.join(", ")}{s.slots.length ? ` · ${s.slots.join("; ")}` : ""}
                                </p>
                            </div>
                        ))}
                    </div>
                )}
            </Section>

            <Section>
                <Reveal>
                    <Eyebrow>Recall</Eyebrow>
                    <Headline className="mb-10">From question to answer — or to “I don’t know.”</Headline>
                </Reveal>
                <StageList stages={RECALL} />
                {ex && (
                    <div className="mt-10 rounded-[28px] bg-cloud p-6 md:p-8">
                        <p className="text-[13px] text-ink-3">Real run · status {ex.recall.status}</p>
                        <p className="text-[21px] font-semibold mt-1 mb-5">“{ex.query}”</p>
                        {ex.recall.memories.map(m => (
                            <div key={m.content} className="py-3 border-t border-line/70">
                                <p className="font-medium">{m.content} <span className="text-accent font-normal">· relevance {m.relevance.toFixed(2)}</span></p>
                                <p className="mt-1 text-[13px] text-ink-2">
                                    {Object.entries(m.explain).filter(([, v]) => v > 0).map(([k, v]) => `${k.replace(/_/g, " ")} ${v.toFixed(2)}`).join(" · ")}
                                </p>
                            </div>
                        ))}
                    </div>
                )}
            </Section>

            <Section tone="night">
                <Reveal>
                    <Eyebrow>Agents</Eyebrow>
                    <Headline>Private or shared. Enforced, not suggested.</Headline>
                    <Lede dark className="mt-5">
                        A private memory is visible only to the agent that wrote it; a shared one to every agent of the same user. The rule is
                        applied in the database on every read — and every user has a separate vector index.
                    </Lede>
                </Reveal>
            </Section>

            <Section tone="cloud">
                <Reveal>
                    <Eyebrow>Learning</Eyebrow>
                    <Headline className="mb-10">It improves from your feedback.</Headline>
                </Reveal>
                <StageList stages={LEARNING} />
            </Section>

            <Section>
                <Reveal>
                    <Eyebrow>Background</Eyebrow>
                    <Headline className="mb-10">Quiet work between conversations.</Headline>
                </Reveal>
                <div className="grid gap-5 md:grid-cols-2">
                    {JOBS.map(j => (
                        <div key={j.name} className="rounded-[28px] bg-cloud p-8">
                            <h3 className="type-title mb-2">{j.name}</h3>
                            <p className="text-ink-2">{j.desc}</p>
                            <p className="mt-4 text-[12px] text-ink-3"><code>{j.key}</code> in config.yaml</p>
                        </div>
                    ))}
                </div>
            </Section>

            <Footer note={<p>Results generated {results.generated_at} at commit {results.git_commit}.</p>} />
        </main>
    );
}
