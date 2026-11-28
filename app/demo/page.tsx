"use client";

import { useEffect, useRef, useState } from "react";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { Eyebrow, Headline, Lede, Section } from "@/components/site/ui";
import { DEMO_API } from "@/lib/demo";

/* The public demo: messages go to the real Recallic pipeline running on Modal (deploy/modal/demo.py), under a
   per-tab session id. Nothing is shared between visitors and everything is gone when the demo server sleeps. */

const EXAMPLES = [
    "uff moved to pune last month, new job at infosys hinjewadi",
    "my daughter meera has ballet on tuesdays 4pm",
    "update: ballet got rescheduled to wednesdays",
    "allergic to penicillin, found out the hard way lol",
];
const QUESTIONS = ["Where do I live?", "When is Meera's ballet?", "What am I allergic to?", "What is my brother's name?"];

type Result = { status: string; answer?: string | null; replaced?: string | null; memories: { content: string; answer?: { text?: string } | null }[]; reason?: string; latency_ms?: number };
type Server = "waking" | "ready" | "down";

function sessionId(): string {
    const fresh = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
    try {
        const s = sessionStorage.getItem("recallic-demo-session") || fresh();
        sessionStorage.setItem("recallic-demo-session", s);
        return s;
    } catch {
        return fresh();
    }
}

async function call(path: string, body?: object) {
    const r = await fetch(`${DEMO_API}${path}`, body
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
        : { method: "GET" });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : `The demo server answered ${r.status}.`);
    return data;
}

export default function DemoPage() {
    const [server, setServer] = useState<Server>("waking");
    const [messages, setMessages] = useState<string[]>([]);
    const [draft, setDraft] = useState("");
    const [question, setQuestion] = useState("");
    const [result, setResult] = useState<Result | null>(null);
    const [asked, setAsked] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const session = useRef("");

    useEffect(() => {
        session.current = sessionId();
        let cancelled = false;
        (async () => {
            for (let i = 0; i < 30 && !cancelled; i++) {          // a cold start takes up to about a minute
                try { await call("/health"); if (!cancelled) setServer("ready"); return; }
                catch { await new Promise((r) => setTimeout(r, 3000)); }
            }
            if (!cancelled) setServer("down");
        })();
        return () => { cancelled = true; };
    }, []);

    async function store(texts: string[]) {
        setBusy(true); setError("");
        try {
            for (const t of texts) {
                await call("/store", { session: session.current, content: t });
                setMessages((m) => [...m, t]);
            }
        } catch (e) { setError((e as Error).message); }
        setBusy(false);
    }

    async function ask(q: string) {
        if (!q.trim()) return;
        setBusy(true); setError(""); setAsked(q);
        try { setResult(await call("/recall", { session: session.current, query: q })); }
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
                <p className="mt-6 text-[14px] text-ink-2" aria-live="polite">
                    {server === "waking" && "Waking the demo server (about 30 seconds the first time)…"}
                    {server === "ready" && "Demo server ready."}
                    {server === "down" && "The demo server is not answering right now. Please try again in a few minutes."}
                </p>
            </Section>

            <Section tone="cloud">
                <div className="grid gap-8 md:grid-cols-2">
                    <div>
                        <h2 className="text-[22px] font-semibold text-ink">1. Your messages</h2>
                        <p className="mt-2 text-[15px] text-ink-2">Write like you would text an assistant, or use the examples.</p>
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
