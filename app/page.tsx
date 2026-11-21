"use client";

import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { ButtonLink, CodeBlock, CountUp, Eyebrow, Headline, Lede, Reveal, ScrollFade, ScrollWords, ScrollZoom, Section, Stat, Tile } from "@/components/site/ui";
import { AgentsHub, ClosingCTA, MemoryField, MemoryStory } from "@/components/site/story";
import { headline, headlineBenchmark, largestClustered, lift, ms, pct, results } from "@/lib/results";

/* Every number on this page is read from eval/results/summary.json (eval/publish_results.py). */

const quickstart = `import os
from memoryos import Memory

mem = Memory(api_key=os.environ["RECALLIX_API_KEY"])

mem.store("The Atlas launch moved to July 14.")
out = mem.recall("When does Atlas launch?")

out["status"]                 # "CONSISTENT" or "ABSTAINED"
out["memories"][0]["answer"]  # "July 14"
out["memories"][0]["explain"] # why it ranked first`;

/** A real recall from the evaluation run, rendered like a terminal session. */
function HeroTerminal() {
    const ex = headline?.example;
    const top = ex?.recall?.memories?.[0];
    return (
        <div className="mx-auto max-w-[820px] text-left rounded-[18px] bg-card overflow-hidden">
            <div className="flex items-center gap-2 px-4 h-10 border-b border-line">
                <span className="h-3 w-3 rounded-full bg-[#ff5f57]" /><span className="h-3 w-3 rounded-full bg-[#febc2e]" /><span className="h-3 w-3 rounded-full bg-[#28c840]" />
                <span className="ml-3 font-mono text-[12px] text-ink-3">recallix — python</span>
            </div>
            <pre className="p-5 md:p-6 text-[13px] leading-[1.7] overflow-x-auto">
                <code>
                    <span className="text-ink-3">import</span> os{"\n"}
                    <span className="text-ink-3">from</span> memoryos <span className="text-ink-3">import</span> Memory{"\n"}
                    mem = Memory(api_key=os.environ[<span className="text-good">&quot;RECALLIX_API_KEY&quot;</span>]){"\n\n"}
                    {ex?.stores?.slice(0, 3).map(s => (
                        <span key={s.content}>mem.store(<span className="text-good">&quot;{s.content}&quot;</span>){"\n"}</span>
                    ))}
                    {ex && <>{"\n"}out = mem.recall(<span className="text-good">&quot;{ex.query}&quot;</span>){"\n"}</>}
                    {ex && <><span className="text-ink-3"># status </span><span className="text-good">{ex.recall.status}</span>{"\n"}</>}
                    {top && <><span className="text-ink-3"># top    </span><span className="text-ink">{top.content}</span>{"  "}<span className="text-accent">relevance {top.relevance.toFixed(2)}</span></>}
                </code>
            </pre>
            <div className="px-5 md:px-6 py-3 border-t border-line font-mono text-[11.5px] text-ink-3">Output from the evaluation run — not a mock-up.</div>
        </div>
    );
}

function FactTimeline() {
    const steps = [
        { when: "March", text: "Project Alpha deadline is December 12.", state: "superseded" },
        { when: "June", text: "The deadline was moved to December 18.", state: "current" },
        { when: "July", text: "The deadline is not December 12 anymore.", state: "negation" },
    ];
    return (
        <div className="grid gap-4 md:grid-cols-3">
            {steps.map((s, i) => (
                <Reveal key={s.when} from={(["left", "up", "right"] as const)[i]} delay={i * 0.12}>
                    <div className={`rounded-[22px] p-6 h-full ${s.state === "current" ? "bg-ink text-black" : "bg-card text-white"}`}>
                        <div className={`type-caption mb-3 ${s.state === "current" ? "text-[#0066cc]" : "text-night-ink-2"}`}>
                            {s.when} · {s.state === "current" ? "kept" : s.state === "negation" ? "invalidates Dec 12 only" : "superseded"}
                        </div>
                        <p className={`text-[19px] leading-snug font-medium ${s.state === "superseded" ? "line-through decoration-1 opacity-60" : ""}`}>{s.text}</p>
                    </div>
                </Reveal>
            ))}
        </div>
    );
}

function ExplainBars() {
    const m = headline?.example?.recall?.memories?.[0];
    if (!m) return null;
    const entries = Object.entries(m.explain).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, 6);
    return (
        <Tile>
            <p className="type-caption text-ink-3 mb-1">Question</p>
            <p className="text-[19px] font-semibold mb-5">“{headline?.example?.query}”</p>
            <p className="type-caption text-ink-3 mb-1">Top memory · relevance {m.relevance.toFixed(2)}</p>
            <p className="text-[17px] mb-7">{m.content}</p>
            <div className="space-y-3">
                {entries.map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[120px_1fr_44px] items-center gap-3 text-[13px]">
                        <span className="text-ink-2">{k.replace(/_/g, " ")}</span>
                        <div className="h-1.5 rounded-full bg-cloud overflow-hidden">
                            <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(1, v) * 100}%` }} />
                        </div>
                        <span className="text-right tabular-nums text-ink-2">{v.toFixed(2)}</span>
                    </div>
                ))}
            </div>
            <p className="type-caption text-ink-3 mt-6">A real recall from the evaluation run, not a mock-up.</p>
        </Tile>
    );
}

export default function HomePage() {
    const bench = headlineBenchmark();
    const gain = lift(bench?.recallix, bench?.dense);
    const ex = headline?.example;
    const story = ex && {
        query: ex.query, status: ex.recall.status,
        memories: ex.stores.map(s => ({
            content: s.content, importance: s.importance, schemas: s.schemas,
            relevance: ex.recall.memories.find(m => m.content === s.content)?.relevance ?? 0,
        })),
    };

    return (
        <main>
            <Nav />

            {/* Hero */}
            <Section hero className="text-center" backdrop={<MemoryField />}>
                <ScrollFade>
                <Reveal from="fade"><p className="mb-3 text-[17px] font-semibold text-ink-2">Recallix</p></Reveal>
                <Headline as="h1">
                    <Reveal as="span" delay={0.08}>Memory for AI agents.</Reveal>
                    <Reveal as="span" delay={0.22} className="text-sheen">Without the guesswork.</Reveal>
                </Headline>
                <Reveal delay={0.36}>
                    <Lede className="mx-auto mt-5 max-w-[560px]">
                        Store what users tell any of their agents. Recall it by meaning. Keep facts current.
                        And decline when nothing answers — without calling an LLM.
                    </Lede>
                </Reveal>
                <Reveal delay={0.48}>
                    <div className="mt-8 flex flex-wrap items-center justify-center gap-6">
                        <ButtonLink href="/sdk">Get started</ButtonLink>
                        <ButtonLink href="/benchmark" variant="link">See the measurements</ButtonLink>
                    </div>
                </Reveal>
                </ScrollFade>
                <ScrollZoom className="mt-16 md:mt-20"><Reveal from="scale" delay={0.6}><HeroTerminal /></Reveal></ScrollZoom>
            </Section>

            {/* Statement — words light up as you scroll */}
            <Section>
                <ScrollWords className="type-headline max-w-[900px]"
                    text="Recallix remembers what your users tell their agents, keeps every fact current, and says “I don’t know” when nothing it holds is an answer." />
            </Section>

            {/* Pinned, scroll-scrubbed walk through one real run */}
            {story && <MemoryStory query={story.query} status={story.status} memories={story.memories} />}

            {/* Measured, not claimed */}
            <Section tone="cloud">
                <Reveal>
                    <Eyebrow tone="muted">Measured, not claimed</Eyebrow>
                    <Headline className="max-w-[760px]">The numbers come from public benchmarks.</Headline>
                    <Lede className="mt-5">
                        {bench ? <>Headline results are from {bench.name}, {bench.questions} questions it was never tuned on.</> : null}
                        {" "}Every figure on this site is generated by the evaluation tools in the repository.
                    </Lede>
                </Reveal>
                <div className="mt-14 grid gap-5 md:grid-cols-2">
                    {bench && <><Reveal from="left"><Tile>
                        <Stat value={gain !== undefined ? <CountUp value={`+${Math.round(gain * 100)}%`} /> : "—"} label="more evidence found than vector search"
                            sub={bench ? `recall@10 ${pct(bench.recallix, 1)} vs ${pct(bench.dense, 1)} with the same embeddings (BM25: ${pct(bench.bm25, 1)}).` : undefined} />
                    </Tile></Reveal>
                    <Reveal from="right" delay={0.08}><Tile>
                        <Stat value={<CountUp value={pct(bench?.declined)} />} label="of unanswerable questions declined"
                            sub={bench ? `while wrongly declining ${pct(bench.wronglyDeclined, 1)} of answerable ones (${bench.nUnanswerable} unanswerable questions).` : undefined} />
                    </Tile></Reveal></>}
                    <Reveal from="left" delay={0.12}><Tile>
                        <Stat value={largestClustered ? <CountUp value={ms(largestClustered.search_avg_ms, 2)} /> : "—"} label={`vector search at ${largestClustered ? (largestClustered.n / 1e6).toFixed(0) + " million" : "—"} memories`}
                            sub={largestClustered ? `recall@10 ${pct(largestClustered.recall_at_10, 1)} against exact search, one CPU thread.` : undefined} />
                    </Tile></Reveal>
                    <Reveal from="right" delay={0.2}><Tile>
                        <Stat value="0" label="LLM calls to store or recall"
                            sub="Facts, importance and ranking are rules and small fitted models. You only need an embedding model." />
                    </Tile></Reveal>
                </div>
            </Section>

            {/* Facts stay current */}
            <Section tone="night">
                <Reveal>
                    <Eyebrow>Truth over time</Eyebrow>
                    <Headline className="max-w-[760px]">Facts change. Recallix keeps up.</Headline>
                    <Lede dark className="mt-5">
                        When a fact is restated, the newer value wins. When something is denied, only the denied value goes —
                        so “not December 12” never erases December 18.
                    </Lede>
                </Reveal>
                <div className="mt-14"><FactTimeline /></div>
            </Section>

            {/* One memory layer, many agents */}
            <Section tone="cloud">
                <div className="grid gap-12 md:grid-cols-[0.9fr_1.1fr] md:items-center">
                    <Reveal from="left">
                        <Eyebrow>Multi-agent</Eyebrow>
                        <Headline>One memory layer. Every agent.</Headline>
                        <Lede className="mt-5">
                            All of a user’s agents read and write the same memory. Mark a memory shared and every agent sees it;
                            keep it private and only the agent that wrote it can.
                        </Lede>
                    </Reveal>
                    <Reveal from="scale" delay={0.1}><AgentsHub /></Reveal>
                </div>
            </Section>

            {/* Reads before it answers */}
            <Section>
                <div className="grid gap-12 md:grid-cols-2 md:items-center">
                    <Reveal from="left">
                        <Eyebrow>Answer check</Eyebrow>
                        <Headline>It reads before it answers.</Headline>
                        <Lede className="mt-5">
                            A small reading-comprehension model, trained to recognise questions a passage cannot answer,
                            checks the memories Recallix is about to return. If none of them answers, you get nothing — not a guess.
                        </Lede>
                        <p className="mt-5 text-[15px] text-ink-2 max-w-[520px]">
                            It runs locally on the CPU. The trade-off between declining and answering is a single, documented setting.
                        </p>
                    </Reveal>
                    <Reveal from="right" delay={0.12}><ExplainBars /></Reveal>
                </div>
            </Section>

            {/* Fast, local */}
            <Section tone="cloud" className="text-center">
                <Reveal>
                    <Eyebrow>Engine</Eyebrow>
                    <Headline>Fast. Local. Yours.</Headline>
                    <Lede className="mx-auto mt-5">
                        A C++ vector engine keeps one index per user, accelerated with NEON on Apple silicon and AVX2 on Intel,
                        and rebuilds itself from SQLite whenever it restarts.
                    </Lede>
                </Reveal>
                <div className="mt-14 grid gap-5 md:grid-cols-3 text-left">
                    {[
                        { t: "Matches the reference", d: "Search quality is measured against exact search and the reference HNSW implementation on identical data." },
                        { t: "Searchable instantly", d: "New memories are found the moment they are written, while the index links them in the background." },
                        { t: "Runs offline", d: "Embeddings can run in-process. No API key, no network, no per-call cost." },
                    ].map((f, i) => (
                        <Reveal key={f.t} from={i === 1 ? "scale" : i === 0 ? "left" : "right"} delay={i * 0.12}><Tile className="h-full">
                            <h3 className="type-title mb-3">{f.t}</h3>
                            <p className="text-ink-2">{f.d}</p>
                        </Tile></Reveal>
                    ))}
                </div>
            </Section>

            {/* Privacy */}
            <Section>
                <Reveal>
                    <Eyebrow>Privacy</Eyebrow>
                    <Headline className="max-w-[760px]">Private by design.</Headline>
                </Reveal>
                <div className="mt-14 grid gap-10 md:grid-cols-3">
                    {[
                        { t: "Keys you can trust", d: "API keys are stored only as hashes, with roles for users, services and admins." },
                        { t: "Nothing leaks across", d: "Every user has a separate index. Private memories are visible only to the agent that wrote them — enforced in the database." },
                        { t: "Delete means delete", d: "Removing a memory removes its vector, its long-term copy and its timeline entry." },
                    ].map((f, i) => (
                        <Reveal key={f.t} delay={i * 0.14}>
                            <h3 className="type-title mb-3">{f.t}</h3>
                            <p className="text-ink-2">{f.d}</p>
                        </Reveal>
                    ))}
                </div>
            </Section>

            {/* Developers */}
            <Section tone="cloud">
                <div className="grid gap-12 md:grid-cols-2 md:items-center">
                    <Reveal from="left">
                        <Eyebrow>Developers</Eyebrow>
                        <Headline>A few lines to remember.</Headline>
                        <Lede className="mt-5">Python and TypeScript SDKs, a small REST API, and one config file.</Lede>
                        <div className="mt-8 flex flex-wrap gap-4 items-center">
                            <ButtonLink href="/sdk">Read the docs</ButtonLink>
                            <ButtonLink href="/flow-control" variant="link">How it works</ButtonLink>
                        </div>
                    </Reveal>
                    <Reveal from="right" delay={0.12}><CodeBlock code={quickstart} label="Python" /></Reveal>
                </div>
            </Section>

            {/* Honest limits */}
            <Section className="text-center">
                <Reveal from="scale">
                    <Eyebrow tone="muted">Honest limits</Eyebrow>
                    <Headline className="mx-auto max-w-[760px]">What it doesn’t do yet.</Headline>
                    <Lede className="mx-auto mt-5">
                        Fact extraction is rule-based and misses some phrasings. Questions that mix up who said what are still
                        often answered instead of declined. The evaluation sets we wrote ourselves are small.
                        We publish all of it.
                    </Lede>
                    <div className="mt-8"><ButtonLink href="/roadmap" variant="link">See the roadmap</ButtonLink></div>
                </Reveal>
            </Section>

            {/* Closing call to action */}
            <ClosingCTA />

            <Footer note={<p>Results generated {results.generated_at} at commit {results.git_commit}.</p>} />
        </main>
    );
}
