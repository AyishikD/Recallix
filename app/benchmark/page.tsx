"use client";

import React from "react";
import Link from "next/link";
import { ArrowLeft, Cpu, Brain, FlaskConical, AlertTriangle } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { evalRuns, hnsw, hnswReference, results, ms, pct, largestClustered } from "@/lib/results";

/* Every number on this page is read from eval/results/summary.json
   (python eval/publish_results.py). Nothing is typed in by hand. */

const evalRows = evalRuns;

function Section({ icon: Icon, title, children, note }: { icon: React.ComponentType<{ className?: string }>; title: string; children: React.ReactNode; note?: string }) {
    return (
        <section className="mb-16">
            <div className="flex items-center gap-3 mb-2">
                <Icon className="w-5 h-5 text-cyan-400" />
                <h2 className="text-2xl font-black tracking-tight">{title}</h2>
            </div>
            {note && <p className="text-sm text-zinc-500 mb-6 max-w-3xl">{note}</p>}
            {children}
        </section>
    );
}

function Table({ head, rows }: { head: string[]; rows: (string | number | undefined)[][] }) {
    return (
        <div className="overflow-x-auto rounded-2xl border border-white/5">
            <table className="w-full text-sm">
                <thead className="bg-white/5 text-[10px] uppercase tracking-widest text-zinc-500">
                    <tr>{head.map(h => <th key={h} className="text-left px-4 py-3 font-black">{h}</th>)}</tr>
                </thead>
                <tbody>
                    {rows.map((r, i) => (
                        <tr key={i} className="border-t border-white/5">
                            {r.map((c, j) => <td key={j} className="px-4 py-3 font-mono text-zinc-300 whitespace-nowrap">{c}</td>)}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default function BenchmarkPage() {
    const chart = hnsw.filter(r => r.n === 100000 && r.dim === 128).map(r => ({
        name: `${r.data} ef=${r.efSearch}`,
        recallix: +(r.recall_at_10 * 100).toFixed(1),
        hnswlib: +((hnswReference.find(h => h.data === r.data && h.efSearch === r.efSearch && h.n === r.n)?.recall_at_10 ?? 0) * 100).toFixed(1),
    }));
    const m = results.machine;

    return (
        <div className="min-h-screen bg-black text-white">
            <nav className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-white/5 flex items-center px-6 glass backdrop-blur-2xl gap-4">
                <Link href="/" className="text-zinc-400 hover:text-white"><ArrowLeft className="w-4 h-4" /></Link>
                <span className="text-sm font-black tracking-tight">MEASURED RESULTS</span>
            </nav>

            <main className="pt-24 pb-24 container mx-auto px-6 max-w-6xl">
                <h1 className="text-5xl md:text-7xl font-black tracking-tighter mb-4">Benchmarks</h1>
                <p className="text-zinc-500 font-bold max-w-3xl mb-12">
                    Generated {results.generated_at} from commit <code>{results.git_commit}</code> on {m.cpu || m.machine} ({m.system}).
                    Reproduce with the commands in <code>eval/README.md</code>. The C++ numbers are single-threaded and exclude the
                    Python API and the embedding model.
                </p>

                <Section icon={Cpu} title="Vector engine" note="Recall@10 is measured against exact (brute-force) search on held-out queries from the same distribution. Clustered data resembles real embeddings; uniform random data is the known worst case for HNSW, shown with the reference implementation for comparison.">
                    {largestClustered && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                            {[["Vectors", largestClustered.n.toLocaleString()], ["Recall@10", pct(largestClustered.recall_at_10, 1)],
                              ["Search avg", ms(largestClustered.search_avg_ms, 3)], ["Search p99", ms(largestClustered.search_p99_ms, 3)]].map(([k, v]) => (
                                <div key={k} className="rounded-2xl border border-white/5 bg-zinc-900/40 p-5">
                                    <div className="text-[10px] font-black uppercase tracking-widest text-zinc-500">{k}</div>
                                    <div className="text-2xl font-black">{v}</div>
                                </div>
                            ))}
                        </div>
                    )}
                    <Table head={["impl", "vectors", "dim", "data", "M / efC / efS", "recall@10", "avg", "p99", "build"]}
                        rows={[...hnsw.map(r => [`recallix (${r.simd})`, r.n.toLocaleString(), r.dim, r.data, `${r.M} / ${r.efConstruction} / ${r.efSearch}`,
                                pct(r.recall_at_10, 1), ms(r.search_avg_ms, 3), ms(r.search_p99_ms, 3), `${r.build_seconds.toFixed(1)} s`]),
                               ...hnswReference.map(r => ["hnswlib", r.n.toLocaleString(), r.dim, r.data, `${r.M} / ${r.efConstruction} / ${r.efSearch}`,
                                pct(r.recall_at_10, 1), ms(r.search_avg_ms, 3), "—", `${r.build_seconds.toFixed(1)} s`])]} />
                    {chart.length > 0 && (
                        <div className="h-72 mt-8 rounded-2xl border border-white/5 bg-zinc-900/40 p-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chart}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1f2937" />
                                    <XAxis dataKey="name" stroke="#6b7280" fontSize={11} />
                                    <YAxis stroke="#6b7280" fontSize={11} unit="%" domain={[0, 100]} />
                                    <Tooltip contentStyle={{ backgroundColor: "#09090b", border: "1px solid #1f2937" }} />
                                    <Bar dataKey="recallix" fill="#22d3ee" name="Recallix recall@10" />
                                    <Bar dataKey="hnswlib" fill="#a78bfa" name="hnswlib recall@10" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </Section>

                <Section icon={Brain} title="Memory pipeline" note="Hand-labelled eval data (eval/data). Models are fitted on dev only; test is held out. 'hash' is a deterministic bag-of-words stand-in used in CI; the real-embedder rows are the meaningful ones. Unanswerable questions count as correct only if Recallix abstains.">
                    <Table head={["split", "embedder", "facts F1", "schema acc", "importance AUC", "hit@1", "hit@3", "false abstain", "unanswerable declined", "recall p50"]}
                        rows={evalRows.map(r => [r.split, r.embedder, r.facts?.f1?.toFixed(2), pct(r.schemas?.accuracy), r.importance?.auc?.toFixed(2),
                            pct(r.retrieval?.["hit@1"]), pct(r.retrieval?.["hit@3"]), pct(r.retrieval?.false_abstain_rate), pct(r.retrieval?.abstain_accuracy),
                            ms(r.retrieval?.recall_ms_p50, 1)])} />
                </Section>

                <Section icon={FlaskConical} title="Legacy benchmarks" note="Pre-existing adversarial gauntlet (v1–v3) and 500-turn retention test, ported unchanged and never used for tuning. Earlier versions of this page reported higher scores that depended on answers hardcoded into the code; those were removed.">
                    <Table head={["run", "gauntlet passed", "retention recall@1"]}
                        rows={evalRows.flatMap(r => r.legacy ? [[`${r.split} · ${r.embedder}`,
                            `${r.legacy.gauntlet_passed}/${r.legacy.gauntlet_total}`, `${r.legacy.retention_hits}/${r.legacy.retention_total}`]] : [])} />
                </Section>

                <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-5 flex gap-3 text-sm text-amber-200/80">
                    <AlertTriangle className="w-5 h-5 shrink-0 text-amber-400" />
                    <p>The eval sets are small (tens of items per task) and were written by the same author as the extraction rules, so
                        they can flatter the system. Independent test data is the most useful contribution — see eval/README.md.</p>
                </div>
            </main>
        </div>
    );
}
