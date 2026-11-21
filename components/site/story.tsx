"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { prefersReducedMotion } from "./motion";

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const easeOut = (x: number) => 1 - Math.pow(1 - x, 3);

/* ------------------------------------------------------------------------------------------
 * MemoryField — a slow, living field of memory points behind the hero (canvas, no library).
 * Pauses when off-screen or in a background tab; draws one still frame under reduced motion.
 * ---------------------------------------------------------------------------------------- */
export function MemoryField({ density = 1 }: { density?: number }) {
    const ref = useRef<HTMLCanvasElement>(null);
    useEffect(() => {
        const canvas = ref.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;
        const reduce = prefersReducedMotion();
        let seed = 11;
        const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        const count = Math.round((window.innerWidth < 640 ? 34 : 70) * density);
        const nodes = Array.from({ length: count }, (_, i) => ({
            x: rnd(), y: rnd(), z: 0.25 + rnd() * 0.75,
            vx: (rnd() - 0.5) * 0.00011, vy: (rnd() - 0.5) * 0.00011,
            accent: i % 11 === 0, phase: rnd() * Math.PI * 2,
        }));
        let w = 0, h = 0, mx = 0, my = 0, tx = 0, ty = 0;
        const resize = () => {
            const dpr = Math.min(2, window.devicePixelRatio || 1);
            w = canvas.clientWidth; h = canvas.clientHeight;
            canvas.width = w * dpr; canvas.height = h * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };
        const draw = (t: number) => {
            ctx.clearRect(0, 0, w, h);
            mx += (tx - mx) * 0.04; my += (ty - my) * 0.04;
            const pts = nodes.map(n => {
                if (!reduce) {
                    n.x += n.vx; n.y += n.vy;
                    if (n.x < -0.05) n.x = 1.05; else if (n.x > 1.05) n.x = -0.05;
                    if (n.y < -0.05) n.y = 1.05; else if (n.y > 1.05) n.y = -0.05;
                }
                return { X: n.x * w + mx * 36 * n.z, Y: n.y * h + my * 36 * n.z, n };
            });
            const maxD = Math.min(170, Math.max(90, w * 0.13));
            ctx.lineWidth = 1;
            for (let i = 0; i < pts.length; i++) {
                for (let j = i + 1; j < pts.length; j++) {
                    const dx = pts[i].X - pts[j].X, dy = pts[i].Y - pts[j].Y;
                    const d2 = dx * dx + dy * dy;
                    if (d2 > maxD * maxD) continue;
                    const a = (1 - Math.sqrt(d2) / maxD) * 0.16 * Math.min(pts[i].n.z, pts[j].n.z);
                    ctx.strokeStyle = `rgba(244,237,224,${a.toFixed(3)})`;
                    ctx.beginPath(); ctx.moveTo(pts[i].X, pts[i].Y); ctx.lineTo(pts[j].X, pts[j].Y); ctx.stroke();
                }
            }
            for (const { X, Y, n } of pts) {
                const pulse = 0.6 + 0.4 * Math.sin(t * 0.0011 + n.phase);
                ctx.fillStyle = n.accent
                    ? `rgba(255,107,91,${(0.6 + 0.4 * pulse).toFixed(3)})`
                    : `rgba(244,237,224,${((0.18 + 0.5 * n.z) * pulse).toFixed(3)})`;
                ctx.beginPath(); ctx.arc(X, Y, (n.accent ? 1.6 : 0.9) + n.z * 1.4, 0, Math.PI * 2); ctx.fill();
            }
        };
        let raf = 0, running = false, visible = true;
        const loop = (t: number) => { draw(t); raf = requestAnimationFrame(loop); };
        const start = () => { if (!running && !reduce && visible && !document.hidden) { running = true; raf = requestAnimationFrame(loop); } };
        const stop = () => { running = false; cancelAnimationFrame(raf); };
        const onResize = () => { resize(); if (!running) draw(0); };
        const onMove = (e: PointerEvent) => { tx = e.clientX / window.innerWidth - 0.5; ty = e.clientY / window.innerHeight - 0.5; };
        const onVisibility = () => (document.hidden ? stop() : start());
        resize(); draw(0);
        const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) start(); else stop(); });
        io.observe(canvas);
        window.addEventListener("resize", onResize);
        window.addEventListener("pointermove", onMove, { passive: true });
        document.addEventListener("visibilitychange", onVisibility);
        return () => {
            stop(); io.disconnect();
            window.removeEventListener("resize", onResize);
            window.removeEventListener("pointermove", onMove);
            document.removeEventListener("visibilitychange", onVisibility);
        };
    }, [density]);
    return (
        <canvas ref={ref} aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-full"
            style={{ maskImage: "radial-gradient(ellipse 75% 70% at 50% 45%, #000 35%, transparent 100%)", WebkitMaskImage: "radial-gradient(ellipse 75% 70% at 50% 45%, #000 35%, transparent 100%)" }} />
    );
}

/* ------------------------------------------------------------------------------------------
 * MemoryStory — a pinned, scroll-scrubbed walk through one real store → recall run.
 * Scrolling drives every frame (and reverses it); all data comes from the evaluation run.
 * ---------------------------------------------------------------------------------------- */
export interface StoryMemory { content: string; importance: number; schemas: string[]; relevance: number }

const SLOTS = [
    { x: 22, y: 20 }, { x: 78, y: 24 }, { x: 18, y: 78 }, { x: 80, y: 76 }, { x: 50, y: 92 },
];

/** Progress (0→1) through a tall section whose child is `position: sticky`. */
export function usePinnedProgress<T extends HTMLElement>() {
    const ref = useRef<T>(null);
    const [p, setP] = useState(0);
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        let frame = 0;
        const update = () => {
            frame = 0;
            const r = el.getBoundingClientRect();
            const total = r.height - window.innerHeight;
            setP(total > 0 ? clamp01(-r.top / total) : 1);
        };
        const onScroll = () => { if (!frame) frame = requestAnimationFrame(update); };
        update();
        window.addEventListener("scroll", onScroll, { passive: true });
        window.addEventListener("resize", onScroll);
        return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); cancelAnimationFrame(frame); };
    }, []);
    return [ref, p] as const;
}

export function MemoryStory({ query, status, memories }: { query: string; status: string; memories: StoryMemory[] }) {
    const [ref, p] = usePinnedProgress<HTMLElement>();

    const mems = memories.slice(0, SLOTS.length);
    const top = mems.reduce((best, m, i) => (m.relevance > mems[best].relevance ? i : best), 0);
    const answered = status === "CONSISTENT";
    const seg = (a: number, b: number) => clamp01((p - a) / (b - a));
    const arrive = seg(0.03, 0.22), understand = seg(0.28, 0.45), recall = seg(0.52, 0.7), answer = seg(0.77, 0.92);
    const step = p < 0.25 ? 0 : p < 0.5 ? 1 : p < 0.75 ? 2 : 3;

    const steps = [
        { t: "Store.", d: "Messages from any of a user’s agents arrive as memories." },
        { t: "Understand.", d: "Each gets its facts, a topic and an importance score. Here, size is importance." },
        { t: "Recall.", d: `A question arrives. Every candidate is scored for relevance; the lines show how strongly.` },
        {
            t: answered ? "Answer." : "Decline.",
            d: answered
                ? "A reading model confirms the top memory actually answers. If none had, Recallix would return nothing."
                : "No memory contains an answer, so Recallix returns nothing rather than a guess.",
        },
    ];

    return (
        <section ref={ref} className="relative bg-paper text-ink" style={{ height: "420vh" }}>
            <div className="sticky top-0 flex h-screen items-center overflow-hidden">
                <div className="mx-auto grid w-full max-w-[1080px] gap-8 px-5 md:grid-cols-[0.8fr_1.2fr] md:items-center">
                    <div>
                        <p className="type-eyebrow mb-3 text-accent">One real run</p>
                        <ol className="space-y-5 md:space-y-7">
                            {steps.map((s, i) => (
                                <li key={s.t} className="transition-opacity duration-500" style={{ opacity: i === step ? 1 : 0.28 }}>
                                    <p className="type-title">{s.t}</p>
                                    <p className={`mt-1 text-[15px] md:text-[17px] text-ink-2 ${i === step ? "" : "hidden md:block"}`}>{s.d}</p>
                                </li>
                            ))}
                        </ol>
                        <div className="mt-8 h-[2px] w-40 rounded-full bg-line overflow-hidden">
                            <div className="h-full bg-ink" style={{ width: `${p * 100}%` }} />
                        </div>
                    </div>

                    <div className="relative h-[52vh] md:h-[min(68vh,560px)] rounded-[28px] bg-card">
                        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                            {mems.map((m, i) => {
                                const isTop = i === top;
                                const op = recall * (0.12 + 0.88 * m.relevance) * (answer > 0 && !isTop ? 1 - 0.75 * answer : 1);
                                return (
                                    <line key={i} x1={50} y1={50}
                                        x2={50 + (SLOTS[i].x - 50) * easeOut(recall)} y2={50 + (SLOTS[i].y - 50) * easeOut(recall)}
                                        stroke={isTop && answer > 0 ? "#FF6B5B" : "#F4EDE0"} strokeOpacity={op}
                                        strokeWidth={1 + 3 * m.relevance} vectorEffect="non-scaling-stroke" strokeLinecap="round" />
                                );
                            })}
                        </svg>

                        {/* the question */}
                        <div className="absolute left-1/2 top-1/2 z-10 max-w-[70%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink px-4 py-2 text-center text-[13px] md:text-[15px] font-medium text-on-ink"
                            style={{ opacity: easeOut(recall), transform: `translate(-50%, -50%) scale(${0.8 + 0.2 * easeOut(recall)})` }}>
                            “{query}”
                        </div>

                        {mems.map((m, i) => {
                            const a = easeOut(clamp01(arrive * 1.7 - i * 0.17));
                            const fromX = i % 2 === 0 ? -30 : 130;
                            const x = fromX + (SLOTS[i].x - fromX) * a;
                            const isTop = i === top;
                            const scale = (0.86 + 0.28 * (m.importance / 10) * easeOut(understand)) * (isTop ? 1 + 0.08 * easeOut(answer) : 1);
                            const dim = answer > 0 && !isTop ? 1 - 0.7 * answer : 1;
                            return (
                                <div key={m.content}
                                    className={`absolute w-[42%] md:w-[38%] rounded-[16px] px-3 py-2.5 md:px-4 md:py-3 text-[12px] md:text-[14px] leading-snug transition-colors duration-500 ${isTop && answer > 0.5 ? "bg-ink text-on-ink" : "bg-[#24375F] text-ink"}`}
                                    style={{ left: `${x}%`, top: `${SLOTS[i].y}%`, opacity: a * dim, transform: `translate(-50%, -50%) scale(${scale})` }}>
                                    <p className="font-medium">{m.content}</p>
                                    <p className={`mt-1 text-[11px] md:text-[12px] ${isTop && answer > 0.5 ? "text-[#C2412F]" : "text-ink-2"}`}>
                                        <span style={{ opacity: easeOut(understand) }}>importance {m.importance.toFixed(1)} · {m.schemas[0] ?? "general"}</span>
                                        <span style={{ opacity: easeOut(recall) }}> · relevance {m.relevance.toFixed(2)}</span>
                                    </p>
                                </div>
                            );
                        })}

                        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-3 py-1 text-[12px] text-ink-2"
                            style={{ opacity: easeOut(answer) }}>
                            status <span className={answered ? "text-good" : "text-warn"}>{status}</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}

/* ------------------------------------------------------------------------------------------
 * AgentsHub — several agents reading and writing one memory layer (SVG, SMIL motion).
 * ---------------------------------------------------------------------------------------- */
const AGENTS = [
    { name: "Support agent", x: 105, y: 70 },
    { name: "Planner", x: 495, y: 70 },
    { name: "Coding agent", x: 105, y: 290 },
    { name: "Research agent", x: 495, y: 290 },
];

export function AgentsHub() {
    const cx = 300, cy = 180;
    return (
        <svg viewBox="0 0 600 360" className="w-full h-auto" role="img" aria-label="Four agents connected to one Recallix memory">
            {AGENTS.map((a, i) => {
                const path = `M${a.x},${a.y} L${cx},${cy}`;
                return (
                    <g key={a.name}>
                        <path d={path} strokeWidth="1.5" fill="none" style={{ stroke: "var(--color-line)" }} />
                        <circle r="3.5" fill="#FF6B5B">
                            <animateMotion dur="2.6s" repeatCount="indefinite" begin={`${i * 0.65}s`} path={path} />
                        </circle>
                        <circle r="3" opacity="0.8" style={{ fill: "var(--color-ink)" }}>
                            <animateMotion dur="2.6s" repeatCount="indefinite" begin={`${1.3 + i * 0.65}s`} path={path} keyPoints="1;0" keyTimes="0;1" calcMode="linear" />
                        </circle>
                        <rect x={a.x - 78} y={a.y - 22} width="156" height="44" rx="22" style={{ fill: "var(--color-card)", stroke: "var(--color-line)" }} />
                        <text x={a.x} y={a.y + 5} textAnchor="middle" fontSize="14" fontWeight="500" style={{ fill: "var(--color-ink)" }}>{a.name}</text>
                    </g>
                );
            })}
            <rect x={cx - 96} y={cy - 40} width="192" height="80" rx="24" style={{ fill: "var(--color-ink)" }} />
            <text x={cx} y={cy - 4} textAnchor="middle" fontSize="19" fontWeight="600" style={{ fill: "var(--color-on-ink)" }}>Recallix</text>
            <text x={cx} y={cy + 18} textAnchor="middle" fontSize="12" style={{ fill: "var(--color-on-ink)", opacity: 0.65 }}>one memory per user</text>
        </svg>
    );
}

/* ------------------------------------------------------------------------------------------
 * PinnedSteps — a sticky section: the step list on the left advances with scroll while
 * `visual(active, progress)` on the right changes to match. Scrolling up reverses it.
 * ---------------------------------------------------------------------------------------- */
export function PinnedSteps({ eyebrow, title, steps, visual, vhPerStep = 70, tone = "paper" }: {
    eyebrow?: string; title?: string; steps: { title: string; detail: string }[];
    visual: (active: number, progress: number) => React.ReactNode; vhPerStep?: number; tone?: "paper" | "cloud";
}) {
    const [ref, p] = usePinnedProgress<HTMLElement>();
    const n = steps.length;
    const active = Math.min(n - 1, Math.floor(p * n));
    return (
        <section ref={ref} className={`relative ${tone === "cloud" ? "tone-cream bg-cloud" : "bg-paper"} text-ink`} style={{ height: `${100 + n * vhPerStep}vh` }}>
            <div className="sticky top-0 flex h-screen items-center overflow-hidden">
                <div className="mx-auto grid w-full max-w-[1080px] gap-8 px-5 md:grid-cols-[0.85fr_1.15fr] md:items-center">
                    <div>
                        {eyebrow && <p className="type-eyebrow mb-2 text-accent">{eyebrow}</p>}
                        {title && <h2 className="type-headline mb-6 md:mb-8">{title}</h2>}
                        <ol className="space-y-2.5 md:space-y-3">
                            {steps.map((s, i) => (
                                <li key={s.title} className="transition-all duration-500" style={{ opacity: i === active ? 1 : i < active ? 0.42 : 0.24 }}>
                                    <p className={`font-semibold tracking-tight transition-all duration-500 ${i === active ? "text-[21px] md:text-[24px]" : "text-[16px] md:text-[17px]"}`}>
                                        <span className="mr-3 text-[13px] font-normal tabular-nums text-ink-3">{String(i + 1).padStart(2, "0")}</span>{s.title}
                                    </p>
                                    <div className="grid transition-all duration-500" style={{ gridTemplateRows: i === active ? "1fr" : "0fr" }}>
                                        <p className="overflow-hidden text-[15px] md:text-[16px] text-ink-2 pl-9 max-w-[440px]">{s.detail}</p>
                                    </div>
                                </li>
                            ))}
                        </ol>
                        <div className="mt-7 h-[2px] w-40 rounded-full bg-line overflow-hidden">
                            <div className="h-full bg-ink" style={{ width: `${p * 100}%` }} />
                        </div>
                    </div>
                    <div className="relative">{visual(active, p)}</div>
                </div>
            </div>
        </section>
    );
}

/* ------------------------------------------------------------------------------------------
 * LayerStack — three slabs in 3D that pull apart with `spread` and highlight `active`.
 * ---------------------------------------------------------------------------------------- */
export function LayerStack({ layers, active, spread }: { layers: { name: string; sub: string }[]; active: number; spread: number }) {
    const gap = 46 + 70 * easeOut(clamp01(spread));
    return (
        <div className="relative mx-auto h-[46vh] md:h-[min(64vh,540px)] w-full" style={{ perspective: "1400px" }}>
            <div className="absolute left-1/2 top-1/2" style={{ transformStyle: "preserve-3d", transform: "translate(-50%, -50%) rotateX(58deg) rotateZ(-34deg)" }}>
                {layers.map((l, i) => {
                    const on = i === active;
                    const z = (layers.length - 1 - i) * gap - ((layers.length - 1) * gap) / 2;
                    return (
                        <div key={l.name}
                            className={`absolute left-1/2 top-1/2 w-[260px] h-[260px] md:w-[300px] md:h-[300px] rounded-[34px] transition-[background-color,box-shadow,opacity] duration-500 ${on ? "bg-ink" : "bg-line"}`}
                            style={{
                                transform: `translate(-50%, -50%) translateZ(${z}px)`,
                                boxShadow: on ? "0 18px 40px -12px rgba(20,33,61,0.45)" : "inset 0 0 0 1px rgba(20,33,61,0.06)",
                            }} />
                    );
                })}
            </div>
            <div className="absolute right-0 top-0 bottom-0 hidden md:flex flex-col justify-center gap-[18%] text-right pr-1">
                {layers.map((l, i) => (
                    <div key={l.name} className="transition-opacity duration-500" style={{ opacity: i === active ? 1 : 0.3 }}>
                        <p className="text-[17px] font-semibold">{l.name}</p>
                        <p className="text-[13px] text-ink-2">{l.sub}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ------------------------------------------------------------------------------------------
 * TypeCode — code that writes itself, line by line, when it scrolls into view.
 * Renders complete without JavaScript or under reduced motion.
 * ---------------------------------------------------------------------------------------- */
export function TypeCode({ code, label }: { code: string; label?: string }) {
    const ref = useRef<HTMLPreElement>(null);
    const [shown, setShown] = useState<number | null>(null);
    const lines = code.split("\n");
    useEffect(() => {
        const el = ref.current;
        if (!el || prefersReducedMotion() || !("IntersectionObserver" in window)) return;
        let timer = 0;
        const io = new IntersectionObserver(([e]) => {
            if (!e.isIntersecting) return;
            io.disconnect();
            let i = 0;
            const next = () => { i += 1; setShown(i); if (i < lines.length) timer = window.setTimeout(next, lines[i - 1].trim() ? 90 : 40); };
            setShown(0);
            timer = window.setTimeout(next, 250);
        }, { threshold: 0.35 });
        io.observe(el);
        return () => { io.disconnect(); window.clearTimeout(timer); };
    }, [lines.length]); // eslint-disable-line react-hooks/exhaustive-deps
    const visible = shown === null ? lines.length : shown;
    return (
        <div className="tone-navy rounded-[20px] bg-code overflow-hidden">
            {label && <div className="px-5 py-3 border-b border-white/10 type-caption text-night-ink-2">{label}</div>}
            <pre ref={ref} className="p-5 text-[13px] leading-relaxed whitespace-pre-wrap break-words">
                <code>
                    {lines.map((l, i) => (
                        <span key={i} className="block transition-opacity duration-300" style={{ opacity: i < visible ? 1 : 0 }}>
                            {l || " "}{i === visible - 1 && shown !== null && shown < lines.length ? <span className="ml-0.5 inline-block w-[7px] h-[14px] -mb-[2px] bg-accent animate-pulse" /> : null}
                        </span>
                    ))}
                </code>
            </pre>
        </div>
    );
}

/** The closing call to action shared by every page. */
export function ClosingCTA() {
    return (
        <section className="relative overflow-hidden bg-night text-ink">
            <MemoryField density={0.7} />
            <div className="relative mx-auto max-w-[1080px] px-5 py-24 md:py-32 text-center">
                <h2 className="type-headline mx-auto max-w-[720px]">Give your agents a memory.</h2>
                <p className="type-lede mx-auto mt-5 max-w-[620px] text-night-ink-2">Open source. Self-hosted. A few lines of code.</p>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-6">
                    <Link href="/sdk" className="inline-flex h-10 items-center rounded-full px-5 text-[15px] font-medium bg-cta text-cta-ink hover:bg-cta-hover transition-colors">Get started</Link>
                    <a href="https://github.com/AyishikD/Recallix" className="text-[17px] text-accent hover:underline">View on GitHub <span aria-hidden="true">›</span></a>
                </div>
            </div>
        </section>
    );
}
