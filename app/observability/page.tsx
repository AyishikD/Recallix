"use client";

import { useCallback, useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { Eyebrow, Headline, Lede, Section } from "@/components/site/ui";
import { API_BASE, memoryApi } from "@/lib/api";

/* Live view of a running Recallix API. Nothing here is sample data: with no API reachable it says so. */

const POLL_MS = 3000;
const HISTORY = 20;

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
    return (
        <div className="rounded-[24px] bg-white p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
            <p className="text-[13px] text-ink-2">{label}</p>
            <p className="mt-2 text-[34px] font-semibold tracking-tight tabular-nums">{value ?? "—"}</p>
        </div>
    );
}

interface Stats { episodic_memories?: number; timeline_events?: number; world_state_keys?: number; reflections?: number }
interface EngineInfo { status?: string; simd?: string; namespaces?: number }
interface SystemInfo { live?: number; pending?: number; M?: number; efSearch?: number; max_level?: number; engine?: EngineInfo }
interface Insight { summary: string; confidence?: number }

export default function DashboardPage() {
    const [stats, setStats] = useState<Stats | null>(null);
    const [sys, setSys] = useState<SystemInfo | null>(null);
    const [insights, setInsights] = useState<Insight[]>([]);
    const [history, setHistory] = useState<{ time: string; memories: number; timeline: number }[]>([]);
    const [reachable, setReachable] = useState<boolean | null>(null);

    const load = useCallback(async () => {
        const [s, y, i] = await Promise.all([memoryApi.stats(), memoryApi.systemStats(), memoryApi.insights()]);
        setReachable(Boolean(s?.data));
        if (s?.data) {
            setStats(s.data);
            setHistory(prev => [...prev, {
                time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
                memories: s.data.episodic_memories ?? 0, timeline: s.data.timeline_events ?? 0,
            }].slice(-HISTORY));
        }
        if (y?.data) setSys({ ...(y.data.user_index || {}), engine: y.data.engine });
        if (i?.data) setInsights(Array.isArray(i.data) ? i.data : []);
    }, []);

    useEffect(() => {
        const first = setTimeout(load, 0);
        const t = setInterval(load, POLL_MS);
        return () => { clearTimeout(first); clearInterval(t); };
    }, [load]);

    return (
        <main>
            <Nav />
            <Section>
                <div className="flex flex-wrap items-end justify-between gap-6">
                    <div>
                        <Eyebrow>Dashboard</Eyebrow>
                        <Headline as="h1">Your memory, live.</Headline>
                    </div>
                    <div className="flex items-center gap-2 text-[14px] text-ink-2 pb-3">
                        <span className={`h-2 w-2 rounded-full ${reachable ? "bg-[#34c759]" : reachable === false ? "bg-[#ff3b30]" : "bg-ink-3"}`} />
                        {reachable ? "Connected" : reachable === false ? "Not connected" : "Connecting"} · <code>{API_BASE}</code>
                    </div>
                </div>
                {reachable === false && (
                    <Lede className="mt-6">
                        No Recallix API answered. Start one locally (see Developers) and build this site with
                        <code className="mx-1">NEXT_PUBLIC_RECALLIX_API_URL</code> and a user key in
                        <code className="mx-1">NEXT_PUBLIC_RECALLIX_API_KEY</code> — never for a public deployment.
                    </Lede>
                )}
            </Section>

            <Section tone="paper">
                <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
                    <Metric label="Memories" value={stats?.episodic_memories} />
                    <Metric label="Vectors indexed" value={sys?.live} />
                    <Metric label="Waiting to link" value={sys?.pending} />
                    <Metric label="Known facts" value={stats?.world_state_keys} />
                    <Metric label="Reflections" value={stats?.reflections} />
                </div>

                <div className="mt-6 grid gap-6 lg:grid-cols-3">
                    <div className="lg:col-span-2 rounded-[28px] bg-white p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)] h-[340px] flex flex-col">
                        <p className="text-[17px] font-semibold">Growth</p>
                        <p className="text-[13px] text-ink-2 mb-4">Memories and timeline events, polled every {POLL_MS / 1000} s.</p>
                        <div className="flex-1">
                            {history.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={history}>
                                        <defs>
                                            <linearGradient id="mem" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#0071e3" stopOpacity={0.18} />
                                                <stop offset="100%" stopColor="#0071e3" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8e8ed" />
                                        <XAxis dataKey="time" stroke="#86868b" fontSize={11} tickLine={false} axisLine={false} />
                                        <YAxis stroke="#86868b" fontSize={11} tickLine={false} axisLine={false} allowDecimals={false} />
                                        <Tooltip contentStyle={{ borderRadius: 12, border: "none", boxShadow: "0 4px 20px rgba(0,0,0,0.08)" }} />
                                        <Area type="monotone" dataKey="memories" name="Memories" stroke="#0071e3" strokeWidth={2} fill="url(#mem)" isAnimationActive={false} />
                                        <Area type="monotone" dataKey="timeline" name="Timeline" stroke="#86868b" strokeWidth={1.5} fill="none" isAnimationActive={false} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-[14px] text-ink-3">Waiting for data</div>
                            )}
                        </div>
                    </div>

                    <div className="rounded-[28px] bg-white p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
                        <p className="text-[17px] font-semibold mb-4">Vector engine</p>
                        <dl className="divide-y divide-line/60 text-[15px]">
                            {[
                                ["Status", sys?.engine?.status ?? "unreachable"],
                                ["SIMD", sys?.engine?.simd],
                                ["User indexes", sys?.engine?.namespaces],
                                ["M", sys?.M],
                                ["efSearch", sys?.efSearch],
                                ["Top layer", sys?.max_level],
                            ].map(([k, v]) => (
                                <div key={k as string} className="flex justify-between py-2.5">
                                    <dt className="text-ink-2">{k}</dt>
                                    <dd className="font-medium tabular-nums">{v ?? "—"}</dd>
                                </div>
                            ))}
                        </dl>
                    </div>
                </div>

                <div className="mt-6 rounded-[28px] bg-white p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
                    <p className="text-[17px] font-semibold mb-4">Reflections</p>
                    {insights.length > 0 ? (
                        <ul className="divide-y divide-line/60">
                            {insights.map((ins, i) => (
                                <li key={i} className="py-3 flex gap-4 justify-between">
                                    <span>{ins.summary}</span>
                                    <span className="shrink-0 text-[13px] text-ink-2 tabular-nums">{Math.round((ins.confidence || 0) * 100)}%</span>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="text-[14px] text-ink-3">None yet. The reflection job writes them from long-term facts (<code>workers.replay_hours</code>).</p>
                    )}
                </div>
            </Section>

            <Footer />
        </main>
    );
}
