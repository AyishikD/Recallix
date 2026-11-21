"use client";

import Link from "next/link";
import { useState } from "react";

export { CountUp, Reveal, ScrollFade, ScrollWords, ScrollZoom } from "./motion";

type Tone = "paper" | "cloud" | "night";
const toneClass: Record<Tone, string> = {
    paper: "bg-paper text-ink",
    cloud: "tone-cream bg-cloud text-ink",
    night: "bg-night text-ink",
};

export function Section({ tone = "paper", children, className = "", id, hero = false, backdrop }:
    { tone?: Tone; children: React.ReactNode; className?: string; id?: string; hero?: boolean; backdrop?: React.ReactNode }) {
    return (
        <section id={id} className={`relative overflow-hidden ${toneClass[tone]} ${className}`}>
            {backdrop}
            <div className={`relative mx-auto max-w-[1080px] px-5 ${hero ? "pt-20 pb-20 md:pt-28 md:pb-28" : "py-20 md:py-28"}`}>{children}</div>
        </section>
    );
}

export function Eyebrow({ children, tone = "accent" }: { children: React.ReactNode; tone?: "accent" | "muted" | "warm" }) {
    const c = tone === "accent" ? "text-accent" : tone === "warm" ? "text-warn" : "text-ink-2";
    return <p className={`type-eyebrow ${c} mb-3`}>{children}</p>;
}

export function Headline({ children, as = "h2", className = "" }: { children: React.ReactNode; as?: "h1" | "h2"; className?: string }) {
    const Tag = as;
    return <Tag className={`${as === "h1" ? "type-hero" : "type-headline"} ${className}`}>{children}</Tag>;
}

export function Lede({ children, className = "", dark = false }: { children: React.ReactNode; className?: string; dark?: boolean }) {
    return <p className={`type-lede ${dark ? "text-night-ink-2" : "text-ink-2"} max-w-[620px] ${className}`}>{children}</p>;
}

export function ButtonLink({ href, children, variant = "primary" }: { href: string; children: React.ReactNode; variant?: "primary" | "secondary" | "link" }) {
    if (variant === "link") {
        return <Link href={href} className="text-[17px] text-accent hover:underline whitespace-nowrap">{children} <span aria-hidden="true">›</span></Link>;
    }
    const cls = variant === "primary"
        ? "bg-cta text-cta-ink font-medium hover:bg-cta-hover"
        : "border border-accent text-accent hover:bg-cta hover:text-cta-ink hover:border-cta";
    return (
        <Link href={href} className={`inline-flex h-10 items-center rounded-full px-5 text-[15px] font-normal transition-colors ${cls}`}>
            {children}
        </Link>
    );
}

/** Large rounded card used for stats and feature summaries. */
export function Tile({ children, className = "", dark = false }: { children: React.ReactNode; className?: string; dark?: boolean }) {
    return (
        <div className={`rounded-[28px] p-8 md:p-10 card-edge text-ink ${dark ? "tone-navy" : ""} ${className}`}>
            {children}
        </div>
    );
}

export function Stat({ value, label, sub, dark = false }: { value: React.ReactNode; label: string; sub?: React.ReactNode; dark?: boolean }) {
    return (
        <div>
            <div className="type-stat whitespace-nowrap">{value}</div>
            <div className={`mt-3 text-[17px] font-semibold text-ink`}>{label}</div>
            {sub && <div className={`mt-1 text-[14px] leading-snug ${dark ? "text-night-ink-2" : "text-ink-2"}`}>{sub}</div>}
        </div>
    );
}

export function CodeBlock({ code, label }: { code: string; label?: string }) {
    const [copied, setCopied] = useState(false);
    return (
        <div className="tone-navy rounded-[20px] bg-code overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-white/10">
                <span className="type-caption text-night-ink-2">{label}</span>
                <button className="type-caption text-night-ink-2 hover:text-white"
                    onClick={() => { navigator.clipboard?.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
                    {copied ? "Copied" : "Copy"}
                </button>
            </div>
            <pre className="p-5 text-[13px] leading-relaxed whitespace-pre-wrap break-words"><code>{code}</code></pre>
        </div>
    );
}

/** Minimal data table in the site's typographic style. */
export function DataTable({ head, rows, dark = false }: { head: string[]; rows: (React.ReactNode)[][]; dark?: boolean }) {
    return (
        <div className={`overflow-x-auto rounded-[20px] card-edge`}>
            <table className="w-full text-[14px]">
                <thead>
                    <tr className={dark ? "text-night-ink-2" : "text-ink-2"}>
                        {head.map(h => <th key={h} className="text-left font-medium px-5 py-4 whitespace-nowrap">{h}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((r, i) => (
                        <tr key={i} className={`border-t ${dark ? "border-white/10" : "border-line/60"}`}>
                            {r.map((c, j) => <td key={j} className={`px-5 py-3.5 whitespace-nowrap tabular-nums ${j === 0 ? "font-medium" : ""}`}>{c}</td>)}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
