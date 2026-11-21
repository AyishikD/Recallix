"use client";

import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { ButtonLink, Eyebrow, Headline, Lede, Reveal, Section, Stat, Tile } from "@/components/site/ui";
import { largestClustered, ms, pct, results } from "@/lib/results";

/* Descriptions follow the code; numbers come from eval/results/summary.json. */

const LAYERS = [
    {
        name: "Memory logic", lang: "Python",
        summary: "Store and recall pipelines built from rules, embeddings and small fitted models. No LLM anywhere in the loop.",
        parts: [
            ["RecallEngine", "Full-text, entity and vector candidates, ranked by a learned logistic model, then answer-checked."],
            ["TruthEngine", "One current value per fact: newer values supersede, “X is not V” removes V only."],
            ["AnswerChecker", "Local SQuAD 2.0 reader. Decides whether any returned memory actually answers."],
            ["Importance", "1 + 9·σ(w·x) over novelty, information content, specificity and cue words."],
        ],
    },
    {
        name: "Vector engine", lang: "C++17",
        summary: "An HNSW index per user with heuristic neighbour selection, shared-mutex concurrency and SIMD distance kernels.",
        parts: [
            ["Per-user indexes", "Namespaces keep every user’s vectors apart; one background thread links new vectors."],
            ["Instant visibility", "Vectors not yet linked into the graph are scanned exactly, so new memories are found at once."],
            ["SIMD", "NEON on ARM64, AVX2 + FMA on x86-64, scalar elsewhere — reported at runtime."],
            ["Shared secret", "The engine refuses to start without RECALLIX_INTERNAL_KEY."],
        ],
    },
    {
        name: "Storage", lang: "SQLite",
        summary: "The system of record, in WAL mode. Every vector index is derived from it and rebuilt automatically after a restart.",
        parts: [
            ["Episodic", "Each memory with its embedding, the embedding model’s name, a content hash and an FTS5 index."],
            ["Long-term", "Promoted facts with Ebbinghaus-style decay."],
            ["Working", "A sliding window of each user’s recent messages."],
            ["Recall log", "Candidate signals per recall — feedback turns them into training data."],
        ],
    },
];

const STACK = [
    ["Python 3.11", "FastAPI, Pydantic, NumPy; optional sentence-transformers and a QA reader"],
    ["C++17", "Own HNSW implementation, NEON / AVX2 intrinsics, cpp-httplib, nlohmann-json"],
    ["SQLite", "WAL mode, FTS5 full-text search, JSON1"],
    ["Next.js", "This site — static export, every number read from the evaluation output"],
];

function FlowDiagram() {
    const box = "rounded-[18px] card-edge px-5 py-4 text-center";
    const arrow = <div className="text-ink-3 text-center text-[20px] leading-none py-1" aria-hidden="true">↓</div>;
    return (
        <div className="max-w-[560px] mx-auto">
            <div className={box}><p className="font-semibold">Your agents</p><p className="text-[13px] text-ink-2">SDK or REST · API key</p></div>
            {arrow}
            <div className={`${box} ring-1 ring-accent/30`}>
                <p className="font-semibold">Recallix API · Python</p>
                <p className="text-[13px] text-ink-2">facts · topics · importance · ranking · truth · answer check</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
                <div>{arrow}<div className={box}><p className="font-semibold">SQLite</p><p className="text-[13px] text-ink-2">system of record</p></div></div>
                <div>{arrow}<div className={box}><p className="font-semibold">Vector engine · C++</p><p className="text-[13px] text-ink-2">one HNSW index per user</p></div></div>
            </div>
            <p className="mt-4 text-center text-[13px] text-ink-3">On start-up the API rebuilds every index from SQLite.</p>
        </div>
    );
}

export default function ArchitecturePage() {
    return (
        <main>
            <Nav />
            <Section hero className="text-center">
                <Reveal>
                    <Eyebrow>Architecture</Eyebrow>
                    <Headline as="h1">Three layers.<br />Each does one thing well.</Headline>
                    <Lede className="mx-auto mt-6">
                        Python decides what to remember and what to return. C++ finds nearest neighbours fast. SQLite keeps everything safe.
                    </Lede>
                </Reveal>
            </Section>

            <Section tone="cloud"><Reveal from="scale"><FlowDiagram /></Reveal></Section>

            {LAYERS.map((layer, i) => (
                <Section key={layer.name} tone={i === 1 ? "night" : "paper"}>
                    <div className="grid gap-12 md:grid-cols-[1fr_1.3fr]">
                        <Reveal from={i % 2 === 0 ? "left" : "right"}>
                            <Eyebrow>{layer.lang}</Eyebrow>
                            <Headline>{layer.name}.</Headline>
                            <Lede dark={i === 1} className="mt-5">{layer.summary}</Lede>
                            {i === 1 && largestClustered && (
                                <div className="mt-10 grid gap-8 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                                    <Stat dark value={ms(largestClustered.search_avg_ms, 2)} label="average search"
                                        sub={`${(largestClustered.n / 1e6).toFixed(0)}M vectors, one thread`} />
                                    <Stat dark value={pct(largestClustered.recall_at_10, 1)} label="recall@10" sub="against exact search" />
                                </div>
                            )}
                        </Reveal>
                        <div className="grid gap-4 sm:grid-cols-2">
                            {layer.parts.map(([name, detail], j) => (
                                <Reveal key={name} from={i % 2 === 0 ? "right" : "left"} delay={j * 0.1}>
                                    <Tile dark={i === 1} className="h-full !p-7">
                                        <h3 className="text-[19px] font-semibold mb-2">{name}</h3>
                                        <p className={`text-[15px] ${i === 1 ? "text-night-ink-2" : "text-ink-2"}`}>{detail}</p>
                                    </Tile>
                                </Reveal>
                            ))}
                        </div>
                    </div>
                </Section>
            ))}

            <Section tone="cloud">
                <Reveal>
                    <Eyebrow tone="muted">Built with</Eyebrow>
                    <Headline className="mb-10">The stack.</Headline>
                </Reveal>
                <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
                    {STACK.map(([t, d], i) => (
                        <Reveal key={t} delay={i * 0.1}>
                            <div className="rounded-[24px] card-edge p-6 h-full">
                                <h3 className="text-[19px] font-semibold mb-2">{t}</h3>
                                <p className="text-[14px] text-ink-2">{d}</p>
                            </div>
                        </Reveal>
                    ))}
                </div>
                <div className="mt-12 flex flex-wrap gap-6">
                    <ButtonLink href="/flow-control" variant="link">Follow a message through</ButtonLink>
                    <ButtonLink href="/benchmark" variant="link">See the measurements</ButtonLink>
                </div>
            </Section>

            <Footer note={<p>Results generated {results.generated_at} at commit {results.git_commit}.</p>} />
        </main>
    );
}
