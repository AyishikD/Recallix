"use client";

import { useState } from "react";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { CountUp, Eyebrow, Headline, Lede, Reveal, ScrollFade, Section } from "@/components/site/ui";
import { AgentsHub, ClosingCTA, MemoryField, PinnedSteps } from "@/components/site/story";
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
    { title: "Extract facts", where: "Python", logStage: "Fact Extraction", detail: "A rule grammar turns statements into (entity, relation, value) facts, each recording the rule and the exact words it came from. “The deadline” is resolved to the deadline mentioned earlier. A precision guard skips casual remarks such as “my cat is sleeping”.",
      code: `"I live in Pune"                     -> user::location = Pune      (rule first_person, quote "Pune")\n"The deadline was moved to Dec 18"  -> projectalpha::deadline = Dec 18   (“the deadline” resolved)` },
    { title: "Tag the topic", where: "Python", logStage: "Schema Classification", detail: "The embedding is compared with example sentences for each topic (identity, preference, calendar, …).",
      code: `score_k = 0.5·cos(v, centroid_k) + 0.5·max cos(v, example_kj) + cue bonus` },
    { title: "Score importance", where: "Python", logStage: "Importance Scoring", detail: "Novelty (an exact nearest-neighbour scan, so it is reproducible), information content, specificity and cue words, combined by weights fitted on labelled data.",
      code: `importance = 1 + 9·sigmoid(w · features)` },
    { title: "Store", where: "SQLite", logStage: "Episodic Storage", detail: "Row, embedding and full-text index in SQLite — the system of record — plus a timeline entry.",
      code: `INSERT INTO episodic_events (...); INSERT INTO episodic_fts (...)` },
    { title: "Index", where: "C++ engine", logStage: "Vector Engine", detail: "The vector joins the user's own HNSW index. It is searchable immediately and linked into the graph in the background.",
      code: `POST /add {"namespace": user_id, "id": memory_id, "vector": [...]}` },
];

const RECALL: Stage[] = [
    { title: "Resolve aliases", where: "SQLite", detail: "If a memory says “X is also known as Y”, a question using the alias is rewritten to the canonical name.",
      code: `"What powers 'The Dragon's Breath'?" -> "What powers Galaxy X-77?"` },
    { title: "Find candidates", where: "SQLite + C++", detail: "Full-text and entity matches restricted to this user inside the index, plus the user's vector index, filtered to memories this agent may see. Memories confirmed by earlier feedback for a similar question always join.",
      code: `lexical = fts5 MATCH 'utok:<user> AND (…)'   # one user's rows only\nvector  = engine /search {namespace: user}` },
    { title: "Rank", where: "Python", detail: "A logistic model over explainable signals — meaning, keywords, entity and relation matches, who a statement is about, recency — fitted by cross-validation.",
      code: `relevance = sigmoid(w · features)   # returned per memory as "explain"` },
    { title: "Keep facts current", where: "Python", detail: "Memories stating the same fact are walked in time order: newer values win, and “X is not V” removes V only.",
      code: `Dec 12  ->  moved to Dec 18  ->  "not Dec 12"   =>   Dec 18` },
    { title: "Check the answer", where: "QA model", detail: "A small reading-comprehension model (SQuAD 2.0) reads the memories in rank order, with “I” resolved to whoever said it, and stops at the first that contains an answer. If none does, Recallix declines. The threshold can be calibrated on your own questions, with a certificate.",
      code: `margin = best answer span − “no answer”\nread 1, then 2 at a time; stop at margin ≥ threshold\nno memory passes  =>  status ABSTAINED` },
    { title: "Check who it’s about", where: "QA model", detail: "If the question names one of the people in the conversation, the reader asks it again about each other person. A memory that “answers” just as well for someone else is not evidence about this person.",
      code: `contrast = margin(“…Bob…”) − max margin(“…Alice…”, …)\ncontrast ≤ 0  =>  not about Bob; keep looking or decline` },
    { title: "Write the receipt", where: "Python", detail: "Every recall explains its decision and the versions involved; every memory carries who said it, when, the rule and quote behind each fact, and how likely each fact is to still be true.",
      code: `receipt.reason   = "answer found in memory 42: \"peanuts\" …"\nfact.freshness   = {still_true: 0.83, half_life_days: 410}` },
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
        <div className="rounded-[28px] card-edge divide-y divide-line/60 overflow-hidden">
            {stages.map((s, i) => (
                <Reveal key={s.title} from={i % 2 === 0 ? "left" : "right"} delay={Math.min(i, 6) * 0.05}>
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
                </Reveal>
            ))}
        </div>
    );
}

/** Right-hand panel of a pinned pipeline: the active stage's code, timing and position. */
function stagePanel(stages: Stage[], timings?: Record<string, number>) {
    return function Panel(active: number) {
        const s = stages[active];
        const t = timings && s.logStage ? timings[s.logStage] : undefined;
        return (
            <div className="tone-navy rounded-[28px] bg-paper p-6 md:p-8 min-h-[340px] md:min-h-[400px] flex flex-col shadow-[0_30px_60px_-30px_rgba(20,33,61,0.6)]">
                <div className="flex items-center justify-between text-[13px] text-ink-2">
                    <span className="tabular-nums">{String(active + 1).padStart(2, "0")} / {String(stages.length).padStart(2, "0")}</span>
                    <span className="rounded-full bg-white/10 px-3 py-1">{s.where}</span>
                </div>
                <div key={active} className="swap-in flex-1">
                    <p className="mt-6 type-title">{s.title}</p>
                    {t !== undefined && <p className="mt-1 text-[15px] text-ink-2"><CountUp value={ms(t, 2)} duration={0.6} /> on average</p>}
                    <pre className="mt-6 text-[13px] leading-relaxed bg-code rounded-[14px] p-4 whitespace-pre-wrap break-words">{s.code}</pre>
                </div>
                <div className="mt-6 flex gap-1.5" aria-hidden="true">
                    {stages.map((_, i) => <span key={i} className={`h-1 flex-1 rounded-full transition-colors duration-500 ${i <= active ? "bg-cta" : "bg-white/15"}`} />)}
                </div>
            </div>
        );
    };
}

export default function HowItWorksPage() {
    const timings = headline?.retrieval?.store_stage_ms_mean;
    const ex = headline?.example;
    return (
        <main>
            <Nav />
            <Section hero className="text-center" backdrop={<MemoryField />}>
                <ScrollFade>
                    <Reveal from="fade"><Eyebrow>How it works</Eyebrow></Reveal>
                    <Headline as="h1">
                        <Reveal as="span" delay={0.08}>No LLM.</Reveal>
                        <Reveal as="span" delay={0.2} className="text-sheen">Every step explained.</Reveal>
                    </Headline>
                    <Reveal delay={0.34}>
                        <Lede className="mx-auto mt-6">
                            Storing and recalling are made of small, inspectable steps. Scroll through each one — what it does, and the code that does it.
                        </Lede>
                    </Reveal>
                </ScrollFade>
            </Section>

            <PinnedSteps eyebrow="Store" title="From message to memory." tone="cloud"
                steps={STORE.map(s => ({ title: s.title, detail: s.detail }))} visual={stagePanel(STORE, timings)} />

            <Section tone="cloud">
                <Reveal>
                    <Eyebrow tone="muted">What came out</Eyebrow>
                    <Headline className="mb-10">Four messages, stored.</Headline>
                </Reveal>
                {timings && <p className="-mt-6 mb-8 text-[13px] text-ink-3">Stage times above are averages measured over the evaluation run, including the embedding model.</p>}
                {ex && (
                    <div className="mt-10 grid gap-3 md:grid-cols-2">
                        {ex.stores.map((s, i) => (
                            <Reveal key={s.content} from={i % 2 === 0 ? "left" : "right"} delay={0.05}><div className="rounded-[20px] card-edge p-5">
                                <p className="font-medium">{s.content}</p>
                                <p className="mt-2 text-[13px] text-ink-2">
                                    importance {s.importance} · {s.schemas.join(", ")}{s.slots.length ? ` · ${s.slots.join("; ")}` : ""}
                                </p>
                            </div></Reveal>
                        ))}
                    </div>
                )}
            </Section>

            <PinnedSteps eyebrow="Recall" title="From question to answer — or to “I don’t know.”"
                steps={RECALL.map(s => ({ title: s.title, detail: s.detail }))} visual={stagePanel(RECALL)} />

            <Section>
                {ex && (
                    <Reveal from="scale"><div className="rounded-[28px] bg-card p-6 md:p-8">
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
                    </div></Reveal>
                )}
            </Section>

            <Section tone="night">
                <div className="grid gap-12 md:grid-cols-[0.9fr_1.1fr] md:items-center">
                    <Reveal from="left">
                        <Eyebrow>Agents</Eyebrow>
                        <Headline>Private or shared. Enforced, not suggested.</Headline>
                        <Lede dark className="mt-5">
                            A private memory is visible only to the agent that wrote it; a shared one to every agent of the same user. The rule is
                            applied in the database on every read — and every user has a separate vector index.
                        </Lede>
                    </Reveal>
                    <Reveal from="right" delay={0.1}><AgentsHub /></Reveal>
                </div>
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
                    {JOBS.map((j, i) => (
                        <Reveal key={j.name} from={i % 2 === 0 ? "left" : "right"} delay={Math.floor(i / 2) * 0.1}><div className="rounded-[28px] bg-card p-8 h-full">
                            <h3 className="type-title mb-2">{j.name}</h3>
                            <p className="text-ink-2">{j.desc}</p>
                            <p className="mt-4 text-[12px] text-ink-3"><code>{j.key}</code> in config.yaml</p>
                        </div></Reveal>
                    ))}
                </div>
            </Section>

            <ClosingCTA />

            <Footer note={<p>Results generated {results.generated_at} at commit {results.git_commit}.</p>} />
        </main>
    );
}
