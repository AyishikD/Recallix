"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { CountUp, DataTable, Eyebrow, Headline, Lede, Reveal, ScrollFade, ScrollWords, ScrollZoom, Section, Stat, Tile } from "@/components/site/ui";
import { ClosingCTA, MemoryField } from "@/components/site/story";
import { evalRuns, hnsw, hnswReference, largestClustered, locomoRuns, longMemEval, ms, pct, results } from "@/lib/results";

/* Every number on this page is read from eval/results/summary.json (eval/publish_results.py). */

const SYSTEM_LABEL: Record<string, string> = {
    recallix: "Recallix", recallix_ranked: "Recallix (ranking only)", dense: "Vector search", bm25: "BM25",
};

export default function BenchmarkPage() {
    const m = results.machine;
    const chart = hnsw.filter(r => r.n === 100000 && r.dim === 128).map(r => ({
        name: `${r.data}, ef ${r.efSearch}`,
        Recallix: +(r.recall_at_10 * 100).toFixed(1),
        hnswlib: +((hnswReference.find(h => h.data === r.data && h.efSearch === r.efSearch && h.n === r.n)?.recall_at_10 ?? 0) * 100).toFixed(1),
    }));

    return (
        <main>
            <Nav />

            <Section hero className="text-center" backdrop={<MemoryField />}>
                <ScrollFade>
                    <Reveal from="fade"><Eyebrow>Benchmarks</Eyebrow></Reveal>
                    <Headline as="h1">
                        <Reveal as="span" delay={0.08}>Measured.</Reveal>
                        <Reveal as="span" delay={0.2} className="text-sheen">In the open.</Reveal>
                    </Headline>
                    <Reveal delay={0.34}>
                        <Lede className="mx-auto mt-6">
                            Generated {results.generated_at} on {m.cpu || m.machine}. Reproduce every number with the commands in
                            <code className="mx-1">eval/README.md</code>.
                        </Lede>
                    </Reveal>
                </ScrollFade>
            </Section>

            <Section>
                <ScrollWords className="type-headline max-w-[900px]"
                    text="Every number on this page is produced by code in the repository, on held-out data, and can be reproduced on your own machine." />
            </Section>

            {longMemEval && (
                <Section tone="cloud">
                    <Reveal>
                        <Eyebrow>Fresh, held-out test</Eyebrow>
                        <Headline>LongMemEval.</Headline>
                        <Lede className="mt-5">
                            {longMemEval.questions} questions over {longMemEval.turns.toLocaleString()} chat turns, run once after every design
                            decision was frozen. Scored on whether the turns containing the answer are retrieved — no LLM judge.
                        </Lede>
                    </Reveal>
                    <div className="mt-12 grid gap-5 md:grid-cols-2">
                        <Reveal from="left"><Tile><Stat value={<CountUp value={pct(longMemEval.systems.recallix?.overall?.["recall@10"], 1)} />} label="evidence recall@10"
                            sub={`Vector search: ${pct(longMemEval.systems.dense?.overall?.["recall@10"], 1)} · BM25: ${pct(longMemEval.systems.bm25?.overall?.["recall@10"], 1)}`} /></Tile></Reveal>
                        <Reveal from="right" delay={0.08}><Tile><Stat value={<CountUp value={pct(longMemEval.abstention.unanswerable_declined)} />} label="of unanswerable questions declined"
                            sub={`Wrongly declined ${pct(longMemEval.abstention.answerable_wrongly_declined, 1)} of ${longMemEval.abstention.n_answerable} answerable questions.`} /></Tile></Reveal>
                    </div>
                    <Reveal className="mt-8">
                        <DataTable head={["System", "recall@5", "recall@10", "hit@10"]}
                            rows={Object.entries(longMemEval.systems).map(([name, r]) => [SYSTEM_LABEL[name] ?? name,
                                pct(r.overall["recall@5"], 1), pct(r.overall["recall@10"], 1), pct(r.overall["hit@10"], 1)])} />
                    </Reveal>
                </Section>
            )}

            {locomoRuns.length > 0 && (
                <Section>
                    <Reveal>
                        <Eyebrow>Public benchmark</Eyebrow>
                        <Headline>LoCoMo.</Headline>
                        <Lede className="mt-5">
                            Conversations 1–5 were used for fitting. Conversations 6–10 were held out from fitting but evaluated
                            repeatedly during development, so read them as optimistic.
                        </Lede>
                    </Reveal>
                    <Reveal from="right" className="mt-10">
                        <DataTable head={["Split", "System", "recall@5", "recall@10", "hit@10"]}
                            rows={locomoRuns.flatMap(run => Object.entries(run.systems).map(([name, r]) => [run.split ?? "all", SYSTEM_LABEL[name] ?? name,
                                pct(r.overall["recall@5"], 1), pct(r.overall["recall@10"], 1), pct(r.overall["hit@10"], 1)]))} />
                    </Reveal>
                    <p className="mt-5 text-[14px] text-ink-2">
                        {locomoRuns.map(run => `${run.split ?? "All conversations"}: declined ${pct(run.abstention.adversarial_declined)} of ${run.abstention.n_adversarial} adversarial questions, wrongly declined ${pct(run.abstention.answerable_wrongly_declined, 1)} of ${run.abstention.n_answerable} answerable.`).join("  ")}
                    </p>
                </Section>
            )}

            <Section tone="night">
                <Reveal>
                    <Eyebrow>Vector engine</Eyebrow>
                    <Headline>Same recall as the reference.</Headline>
                    <Lede dark className="mt-5">
                        Recall@10 against exact search on held-out queries, single thread, next to the reference HNSW implementation on
                        identical data. Uniform random vectors are the known worst case for any HNSW index.
                    </Lede>
                </Reveal>
                {largestClustered && (
                    <div className="mt-12 grid gap-5 md:grid-cols-3">
                        <Reveal from="left"><Tile dark className="h-full"><Stat dark value={`${(largestClustered.n / 1e6).toFixed(0)}M`} label="vectors" sub={`${largestClustered.dim} dimensions, clustered`} /></Tile></Reveal>
                        <Reveal from="scale" delay={0.1}><Tile dark className="h-full"><Stat dark value={<CountUp value={pct(largestClustered.recall_at_10, 1)} />} label="recall@10" sub="vs exact search" /></Tile></Reveal>
                        <Reveal from="right" delay={0.2}><Tile dark className="h-full"><Stat dark value={<CountUp value={ms(largestClustered.search_avg_ms, 2)} />} label="average search" sub={`p99 ${ms(largestClustered.search_p99_ms, 2)}`} /></Tile></Reveal>
                    </div>
                )}
                {chart.length > 0 && (
                    <ScrollZoom className="mt-8"><div className="h-72 rounded-[20px] bg-night-2 p-5">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chart} barGap={6}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2B3D66" />
                                <XAxis dataKey="name" stroke="#C3C7D1" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="#C3C7D1" fontSize={12} unit="%" domain={[0, 100]} tickLine={false} axisLine={false} />
                                <Tooltip contentStyle={{ background: "#1C2D52", border: "1px solid #2B3D66", borderRadius: 12, color: "#fff" }} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                                <Bar dataKey="Recallix" fill="#FF6B5B" radius={[6, 6, 0, 0]} isAnimationActive={false} />
                                <Bar dataKey="hnswlib" fill="#8D96AB" radius={[6, 6, 0, 0]} isAnimationActive={false} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div></ScrollZoom>
                )}
                <div className="mt-8">
                    <DataTable dark head={["Implementation", "Vectors", "Dim", "Data", "M / efC / efS", "recall@10", "Average", "p99", "Build"]}
                        rows={[...hnsw.map(r => [`Recallix (${r.simd})`, r.n.toLocaleString(), r.dim, r.data, `${r.M} / ${r.efConstruction} / ${r.efSearch}`,
                            pct(r.recall_at_10, 1), ms(r.search_avg_ms, 3), ms(r.search_p99_ms, 3), `${r.build_seconds.toFixed(1)} s`]),
                        ...hnswReference.map(r => ["hnswlib", r.n.toLocaleString(), r.dim, r.data, `${r.M} / ${r.efConstruction} / ${r.efSearch}`,
                            pct(r.recall_at_10, 1), ms(r.search_avg_ms, 3), "—", `${r.build_seconds.toFixed(1)} s`])]} />
                </div>
            </Section>

            <Section tone="cloud">
                <Reveal>
                    <Eyebrow tone="muted">Our own test sets</Eyebrow>
                    <Headline>The small print.</Headline>
                    <Lede className="mt-5">
                        Hand-labelled sets written by the same author as the extraction rules — small, and evaluated repeatedly during
                        development. Useful for regressions, not for bragging. “hash” is a deterministic stand-in embedder used in CI.
                    </Lede>
                </Reveal>
                <div className="mt-10">
                    <DataTable head={["Split", "Embedder", "Facts F1", "Schema", "Importance AUC", "hit@1", "hit@3", "Wrongly declined", "Unanswerable declined", "Gauntlet", "Retention"]}
                        rows={evalRuns.map(r => [r.split, r.embedder.replace("st:sentence-transformers/", ""), r.facts?.f1?.toFixed(2), pct(r.schemas?.accuracy),
                            r.importance?.auc?.toFixed(2), pct(r.retrieval?.["hit@1"]), pct(r.retrieval?.["hit@3"]), pct(r.retrieval?.false_abstain_rate),
                            pct(r.retrieval?.abstain_accuracy), r.legacy ? `${r.legacy.gauntlet_passed}/${r.legacy.gauntlet_total}` : "—",
                            r.legacy ? `${r.legacy.retention_hits}/${r.legacy.retention_total}` : "—"])} />
                </div>
            </Section>

            <ClosingCTA />

            <Footer note={<p>Generated {results.generated_at} at commit {results.git_commit}.</p>} />
        </main>
    );
}
