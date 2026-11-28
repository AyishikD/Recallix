"use client";

import { useEffect, useRef, useState } from "react";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { Eyebrow, Headline, Lede, Section } from "@/components/site/ui";
import { DEMO_API, TURNSTILE_SITE_KEY } from "@/lib/demo";

/* The public demo: the page passes Cloudflare's human check (Turnstile) once, gets a session from the gate
   (deploy/cloudflare/demo-gate) and talks only to it; the gate applies the limits and forwards to the real Recallic
   pipeline on Modal (deploy/modal/demo.py). Nothing is shared between visitors and everything is gone when the demo
   server sleeps. */

declare global {
    interface Window { turnstile?: { render: (el: HTMLElement, opts: Record<string, unknown>) => string; reset: (id?: string) => void; remove: (id: string) => void } }
}

const EXAMPLES = [
    "uff moved to pune last month, new job at infosys hinjewadi",
    "my daughter meera has ballet on tuesdays 4pm",
    "update: ballet got rescheduled to wednesdays",
    "allergic to penicillin, found out the hard way lol",
];
const QUESTIONS = ["Where do I live?", "When is Meera's ballet?", "What am I allergic to?", "What is my brother's name?"];

type Result = { status: string; answer?: string | null; replaced?: string | null; memories: { content: string; answer?: { text?: string } | null }[]; reason?: string; latency_ms?: number };
type Server = "checking" | "tick" | "slow" | "waking" | "ready" | "down";
const TOKEN_KEY = "recallic-demo-token";

function savedToken(): string {
    try {
        const t = sessionStorage.getItem(TOKEN_KEY) || "";
        return Number(t.split(".")[1] || 0) > Date.now() / 1000 + 60 ? t : "";   // still valid for a minute or more
    } catch {
        return "";
    }
}

function saveToken(t: string) {
    try { if (t) sessionStorage.setItem(TOKEN_KEY, t); else sessionStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
}

async function call(path: string, token: string, body?: object) {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const r = await fetch(`${DEMO_API}${path}`, body ? { method: "POST", headers, body: JSON.stringify(body) } : { method: "GET", headers });
    const data = await r.json().catch(() => ({}));
    if (r.status === 401) saveToken("");
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : `The demo answered ${r.status}.`);
    return data;
}

const STATUS: Record<Server, string> = {
    checking: "Getting the demo ready…",
    tick: "One quick check: tick the box below.",
    slow: "This is taking longer than usual.",
    waking: "Waking the demo server, about 20 seconds…",
    ready: "Ready. Try it below.",
    down: "The demo is not answering right now. Please try again in a few minutes.",
};

function StatusBadge({ server, dark = false, boxAbove = false }: { server: Server; dark?: boolean; boxAbove?: boolean }) {
    const busy = server === "checking" || server === "waking";
    const attention = server === "tick" || server === "slow";
    const tone = server === "ready" ? "border-[#2E7D4F] text-[#2E7D4F]" : server === "down" ? "border-[#B42318] text-[#B42318]"
        : attention ? "border-[#FF6B5B] text-[#FF6B5B]"
        : dark ? "border-[#F4EDE0]/50 text-[#F4EDE0]" : "border-[#14213D]/40 text-[#14213D]";
    return (
        <span role="status" className={`inline-flex items-center gap-3 rounded-full border-2 px-5 py-2.5 text-[17px] font-medium ${tone} ${server === "ready" && dark ? "bg-[#F4EDE0] border-[#F4EDE0]" : ""}`}>
            {busy
                ? <span aria-hidden="true" className="h-4 w-4 rounded-full border-2 border-current border-t-transparent animate-spin" />
                : <span aria-hidden="true" className={`h-3 w-3 rounded-full ${server === "ready" ? "bg-[#2E7D4F]" : attention ? "bg-[#FF6B5B]" : "bg-[#B42318]"}`} />}
            {server === "tick" && boxAbove ? "One quick check: tick the box at the top of the page." : STATUS[server]}
        </span>
    );
}

export default function DemoPage() {
    const [server, setServer] = useState<Server>("checking");
    const [messages, setMessages] = useState<string[]>([]);
    const [draft, setDraft] = useState("");
    const [question, setQuestion] = useState("");
    const [result, setResult] = useState<Result | null>(null);
    const [asked, setAsked] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const token = useRef("");
    const check = useRef<HTMLDivElement>(null);
    const widget = useRef<string | undefined>(undefined);
    const [attempt, setAttempt] = useState(0);

    useEffect(() => {
        let cancelled = false;
        const wake = async (t: string) => {
            token.current = t;
            setServer("waking");
            for (let i = 0; i < 30 && !cancelled; i++) {          // a cold start takes up to about a minute
                try { await call("/health", t); if (!cancelled) setServer("ready"); return; }
                catch (e) {
                    if (!savedToken()) { if (!cancelled) { setServer("down"); setError((e as Error).message); } return; }
                    await new Promise((r) => setTimeout(r, 3000));
                }
            }
            if (!cancelled) setServer("down");
        };
        const saved = savedToken();
        if (saved) { wake(saved); return () => { cancelled = true; }; }
        let passed = false;
        const slow = setTimeout(() => { if (!passed && !cancelled) setServer((s) => (s === "checking" ? "slow" : s)); }, 20000);
        const render = () => {
            if (!window.turnstile || !check.current) return;
            widget.current = window.turnstile.render(check.current, {
                sitekey: TURNSTILE_SITE_KEY,
                appearance: "interaction-only",            // invisible unless a click is needed
                theme: "light",                            // stands out on the navy hero when it does show
                "before-interactive-callback": () => { if (!cancelled) setServer("tick"); },
                "after-interactive-callback": () => { if (!cancelled) setServer("checking"); },
                "expired-callback": () => { if (!cancelled) setServer("tick"); },
                "timeout-callback": () => { if (!cancelled) setServer("slow"); },
                callback: async (human: string) => {
                    passed = true;
                    try {
                        const { token: t } = await call("/session", "", { turnstile: human });
                        saveToken(t);
                        if (!cancelled) wake(t);
                    } catch (e) { if (!cancelled) { setServer("down"); setError((e as Error).message); } }
                },
                "error-callback": () => { if (!cancelled) { setServer("down"); setError("The human check could not load. Please reload the page."); } },
            });
        };
        if (window.turnstile) render();
        else {
            const tag = document.createElement("script");
            tag.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
            tag.async = true;
            tag.onload = render;
            tag.onerror = () => { if (!cancelled) { setServer("down"); setError("The check could not load. Is a content blocker on? Please reload."); } };
            document.head.appendChild(tag);
        }
        return () => {
            cancelled = true; clearTimeout(slow);
            if (widget.current && window.turnstile) { try { window.turnstile.remove(widget.current); } catch { /* already gone */ } }
            widget.current = undefined;
        };
    }, [attempt]);

    async function store(texts: string[]) {
        setBusy(true); setError("");
        try {
            for (const t of texts) {
                await call("/store", token.current, { content: t });
                setMessages((m) => [...m, t]);
            }
        } catch (e) { setError((e as Error).message); }
        setBusy(false);
    }

    async function ask(q: string) {
        if (!q.trim()) return;
        setBusy(true); setError(""); setAsked(q);
        try { setResult(await call("/recall", token.current, { query: q })); }
        catch (e) { setError((e as Error).message); }
        setBusy(false);
    }

    const ready = server === "ready";
    const answered = result?.status === "CONSISTENT";
    return (
        <main>
            <Nav />
            <Section hero>
                <Eyebrow>Try it</Eyebrow>
                <Headline as="h1">Tell it a few things. Then ask.</Headline>
                <Lede className="mt-6">This is the real Recallic, no LLM involved. It answers from what you told it, and when nothing answers, it says so instead of guessing.</Lede>
                <div className="mt-8 flex flex-wrap items-center gap-4" aria-live="polite">
                    <StatusBadge server={server} dark />
                    {(server === "slow" || server === "down") && (
                        <button type="button" onClick={() => { setError(""); setServer("checking"); setAttempt((a) => a + 1); }}
                            className="rounded-full bg-cta text-cta-ink px-5 h-11 text-[15px] hover:bg-cta-hover">Try again</button>
                    )}
                </div>
                <div ref={check} className="mt-4" />
            </Section>

            <Section tone="cloud">
                <div className="grid gap-8 md:grid-cols-2">
                    <div>
                        <h2 className="text-[22px] font-semibold text-ink">1. Your messages</h2>
                        <p className="mt-2 text-[15px] text-ink-2">Write like you would text an assistant, or use the examples.</p>
                        {server !== "ready" && <div className="mt-4"><StatusBadge server={server} boxAbove /></div>}
                        <ul className="mt-5 space-y-2">
                            {messages.map((m, i) => (
                                <li key={i} className="rounded-[16px] bg-[#14213D] px-4 py-3 text-[15px] text-[#F4EDE0]">{m}</li>
                            ))}
                            {messages.length === 0 && <li className="text-[14px] text-ink-2">No messages yet.</li>}
                        </ul>
                        <form className="mt-4 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (draft.trim()) { store([draft.trim()]); setDraft(""); } }}>
                            <label htmlFor="msg" className="sr-only">A message to remember</label>
                            <input id="msg" value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={400}
                                placeholder="e.g. my sister mira lives in goa"
                                className="flex-1 min-w-0 rounded-full border border-line bg-white px-4 h-11 text-[15px] text-[#14213D]" />
                            <button type="submit" disabled={!ready || busy || !draft.trim()}
                                className="rounded-full bg-cta text-cta-ink px-5 h-11 text-[15px] disabled:opacity-40">Remember</button>
                        </form>
                        {messages.length === 0 && (
                            <button type="button" disabled={!ready || busy} onClick={() => store(EXAMPLES)}
                                className="mt-3 text-[15px] text-accent hover:underline disabled:opacity-40">Use the example messages ›</button>
                        )}
                    </div>

                    <div>
                        <h2 className="text-[22px] font-semibold text-ink">2. Ask</h2>
                        <p className="mt-2 text-[15px] text-ink-2">Try one that has an answer, and one that does not.</p>
                        <div className="mt-5 flex flex-wrap gap-2">
                            {QUESTIONS.map((q) => (
                                <button key={q} type="button" disabled={!ready || busy || messages.length === 0} onClick={() => { setQuestion(q); ask(q); }}
                                    className="rounded-full border border-accent text-accent px-4 h-10 text-[14px] hover:bg-cta hover:text-cta-ink disabled:opacity-40">{q}</button>
                            ))}
                        </div>
                        <form className="mt-4 flex gap-2" onSubmit={(e) => { e.preventDefault(); ask(question); }}>
                            <label htmlFor="q" className="sr-only">Your question</label>
                            <input id="q" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={400}
                                placeholder="Your own question"
                                className="flex-1 min-w-0 rounded-full border border-line bg-white px-4 h-11 text-[15px] text-[#14213D]" />
                            <button type="submit" disabled={!ready || busy || messages.length === 0 || !question.trim()}
                                className="rounded-full bg-cta text-cta-ink px-5 h-11 text-[15px] disabled:opacity-40">Ask</button>
                        </form>

                        <div className="mt-6" aria-live="polite">
                            {busy && <p className="text-[14px] text-ink-2">Working…</p>}
                            {error && <p className="text-[15px] text-[#B42318]">{error}</p>}
                            {result && !busy && (
                                <div className="rounded-[20px] card-edge p-6">
                                    <p className="text-[14px] text-ink-2">{asked}</p>
                                    {answered ? (
                                        <>
                                            <p className="mt-2 text-[24px] font-semibold text-ink">{result.answer || result.memories[0]?.content}</p>
                                            {result.memories[0] && <p className="mt-3 text-[15px] text-ink-2">From: “{result.memories[0].content}”</p>}
                                            {result.replaced && <p className="mt-1 text-[14px] text-ink-2">Updated from “{result.replaced}”: the newer message wins.</p>}
                                        </>
                                    ) : (
                                        <p className="mt-2 text-[22px] font-semibold text-ink">No record. It declines instead of guessing.</p>
                                    )}
                                    {result.reason && <p className="mt-3 text-[13px] text-ink-2">Receipt: {result.reason}</p>}
                                    {typeof result.latency_ms === "number" && <p className="mt-1 text-[13px] text-ink-2">{Math.round(result.latency_ms)} ms</p>}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                <p className="mt-10 text-[14px] text-ink-2 max-w-[720px]">
                    Your messages are visible only in this browser tab and are erased when the demo server sleeps, a few minutes after the last visitor.
                    Each message counts as one day after the one before, like a week of texting. Please don&apos;t type anything private. Limit: 5 messages per visitor per day, 400 characters each.
                </p>
            </Section>
            <Footer />
        </main>
    );
}
