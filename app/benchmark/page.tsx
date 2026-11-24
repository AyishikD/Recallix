"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { CountUp, DataTable, Eyebrow, Headline, Lede, Reveal, ScrollFade, ScrollWords, ScrollZoom, Section, Stat, Tile } from "@/components/site/ui";
import { ClosingCTA, MemoryField } from "@/components/site/story";
import { evalRuns, hnsw, hnswReference, largestClustered, locomoRuns, longMemEval, longMemEvalDeep, mem0Comparison, ms, pct, perMillion, results, writeSpeedup } from "@/lib/results";

const MEM0 = mem0Comparison;
const usd = (x: number) => `$${x.toFixed(2)}`;
const n = (x: number) => Math.round(x).toLocaleString("en-US");

/* Every number on this page is read from eval/results/summary.json (eval/publish_results.py). */

const SYSTEM_LABEL: Record<string, string> = {
    recallix: "Recallix (served, after answer check)", recallix_ranked: "Recallix ranking", dense: "Vector search", bm25: "BM25",
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
                            {longMemEval.questions} questions over {longMemEval.turns.toLocaleString()} chat turns. Never used to fit or choose
                            anything; each run uses a configuration fixed beforehand (earlier runs stay in the repository). Scored on whether
                            the turns containing the answer are retrieved — no LLM judge.
                        </Lede>
                    </Reveal>
                    <div className="mt-12 grid gap-5 md:grid-cols-2">
                        <Reveal from="left"><Tile><Stat value={<CountUp value={pct(longMemEval.systems.recallix_ranked?.overall?.["recall@10"], 1)} />} label="evidence recall@10, Recallix ranking"
                            sub={`Vector search ${pct(longMemEval.systems.dense?.overall?.["recall@10"], 1)} · BM25 ${pct(longMemEval.systems.bm25?.overall?.["recall@10"], 1)} · served after the answer check ${pct(longMemEval.systems.recallix?.overall?.["recall@10"], 1)}`} /></Tile></Reveal>
                        <Reveal from="right" delay={0.08}><Tile><Stat value={<CountUp value={pct(longMemEval.abstention.unanswerable_declined)} />} label="of unanswerable questions declined"
                            sub={`Wrongly declined ${pct(longMemEval.abstention.answerable_wrongly_declined, 1)} of ${longMemEval.abstention.n_answerable} answerable questions. The answer check separates the two well (AUC ${longMemEval.abstention.auc_answer_margin?.toFixed(2) ?? "—"}), but its threshold was set on LoCoMo and declines too often here; it needs calibrating per deployment.`} /></Tile></Reveal>
                    </div>
                    <Reveal className="mt-8">
                        <DataTable head={["System", "recall@5", "recall@10", "hit@10"]}
                            rows={[...Object.entries(longMemEval.systems).map(([name, r]) => [SYSTEM_LABEL[name] ?? name,
                                pct(r.overall["recall@5"], 1), pct(r.overall["recall@10"], 1), pct(r.overall["hit@10"], 1)]),
                                ...(longMemEvalDeep ? [["Recallix, deep recall (served)",
                                    pct(longMemEvalDeep.systems.recallix.overall["recall@5"], 1), pct(longMemEvalDeep.systems.recallix.overall["recall@10"], 1),
                                    pct(longMemEvalDeep.systems.recallix.overall["hit@10"], 1)]] : [])]} />
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

            {/* Head-to-head with an existing LLM-based memory layer (not named on the site, by choice):
                every number from eval/results/mem0_comparison/summary.json */}
            <Section tone="cloud" id="head-to-head">
                <Reveal>
                    <Eyebrow tone="muted">Head to head</Eyebrow>
                    <Headline className="max-w-[760px]">Recallix vs an existing memory layer, same questions.</Headline>
                    <Lede className="mt-5 max-w-[680px]">
                        Recallix and a widely used open-source memory layer that calls an LLM on every write both stored the same held-out LoCoMo conversation ({n(MEM0.scope.messages)} messages) and answered the
                        same {MEM0.scope.questions} questions. One model wrote every final answer and one judge marked them, so only the
                        memory layer differs. The protocol was written before the run.
                    </Lede>
                </Reveal>
                <Reveal from="right" className="mt-10">
                    <DataTable head={["", "Recallix", "LLM-based memory layer"]} rows={[
                        ["Accuracy, all questions", pct(MEM0.accuracy.recallix.accuracy_overall, 1), pct(MEM0.accuracy.mem0.accuracy_overall, 1)],
                        ["Accuracy, answerable questions", pct(MEM0.accuracy.recallix.accuracy_answerable, 1), pct(MEM0.accuracy.mem0.accuracy_answerable, 1)],
                        ["Trick questions declined", pct(MEM0.accuracy.recallix.adversarial_declined, 1), pct(MEM0.accuracy.mem0.adversarial_declined, 1)],
                        ...["single-hop", "temporal", "multi-hop"].map(c => [`Accuracy, ${c} (${MEM0.scope.per_category[c]} questions)`,
                            pct(MEM0.accuracy.recallix.by_category[c], 1), pct(MEM0.accuracy.mem0.by_category[c], 1)]),
                        ["LLM calls to store the conversation", n(MEM0.store.recallix.llm_calls), n(MEM0.store.mem0.llm_calls)],
                        ["Cost to store the conversation", usd(MEM0.store.recallix.usd), usd(MEM0.store.mem0.usd)],
                        ["Store one message, median", `${n(MEM0.store.recallix.ms_p50)} ms`, `${n(MEM0.store.mem0.ms_p50)} ms`],
                        ["Answer one question, median", `${n(MEM0.recall_ms.recallix.p50)} ms`, `${n(MEM0.recall_ms.mem0.p50)} ms`],
                    ]} />
                </Reveal>
                <div className="mt-6 max-w-[760px] space-y-3 text-[14px] text-ink-2">
                    <p>
                        <strong className="text-ink">Accuracy is a statistical tie.</strong> Recallix alone was right on{" "}
                        {MEM0.significance.all.only_recallix_right} questions, the other system alone on {MEM0.significance.all.only_mem0_right}{" "}
                        (exact McNemar p = {MEM0.significance.all.p.toFixed(2)}). No per-category difference is significant at this sample
                        size. The LLM-based system is stronger on multi-hop questions; Recallix on single facts and dates.
                    </p>
                    <p>
                        <strong className="text-ink">Cost and speed are not close.</strong> Recallix stores without calling an LLM; the other
                        system made {n(MEM0.store.mem0.llm_calls)} model calls to store one conversation. Writes were about {writeSpeedup(MEM0)}×
                        faster at the median. Extrapolated from its measured cost per message, a million messages would cost about
                        ${perMillion(MEM0).toLocaleString("en-US")} in LLM calls to store; Recallix stays at $0.
                    </p>
                    <p className="text-ink-3">
                        One conversation, so the error bars are wide. The other system ran its open-source release at default settings,
                        with gpt-4o-mini extracting memories. Recallix: {MEM0.setup.recallix}. Answers and judging:{" "}
                        {MEM0.setup.answer_and_judge}. Recallix was timed in-process on an Apple M4; the other system’s times include its
                        model calls. Its cost is estimated from token usage at list prices. A second conversation was stopped part-way to
                        stay within the test budget.
                    </p>
                </div>
            </Section>

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
