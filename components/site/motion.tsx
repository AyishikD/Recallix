"use client";

import { useEffect, useRef } from "react";

/* Apple-style scroll motion.
 * Progressive enhancement throughout: content is rendered visible, and only once
 * JavaScript runs (and the visitor has not asked for reduced motion) is it hidden
 * and animated in. Nothing here can leave text invisible. */

export const prefersReducedMotion = () =>
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export type RevealFrom = "up" | "left" | "right" | "scale" | "fade";

/** Slides/fades its children in once, when they scroll into view. */
export function Reveal({ children, from = "up", delay = 0, className = "", as = "div" }: {
    children: React.ReactNode; from?: RevealFrom; delay?: number; className?: string; as?: "div" | "span";
}) {
    const ref = useRef<HTMLElement>(null);
    useEffect(() => {
        const el = ref.current;
        if (!el || prefersReducedMotion() || !("IntersectionObserver" in window)) return;
        el.classList.add("reveal-armed");
        const io = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) {
                el.classList.add("reveal-in");
                io.disconnect();
            }
        }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
        io.observe(el);
        return () => io.disconnect();
    }, []);
    const Tag = as;
    return (
        <Tag ref={ref as React.Ref<never>} data-from={from} style={delay ? { transitionDelay: `${delay}s` } : undefined}
            className={`reveal ${as === "span" ? "block" : ""} ${className}`}>
            {children}
        </Tag>
    );
}

/** Runs `onProgress(p)` with p ∈ [0, 1] as the element travels through the viewport. */
export function useScrollProgress<T extends HTMLElement>(onProgress: (p: number, el: T) => void, start = 1, end = 0.35) {
    const ref = useRef<T>(null);
    const cb = useRef(onProgress);
    useEffect(() => { cb.current = onProgress; });
    useEffect(() => {
        const el = ref.current;
        if (!el || prefersReducedMotion()) return;
        let frame = 0;
        const update = () => {
            frame = 0;
            const r = el.getBoundingClientRect();
            const vh = window.innerHeight;
            // 0 when the element's top is at `start`·vh, 1 when it reaches `end`·vh
            const p = (start * vh - r.top) / ((start - end) * vh);
            cb.current(Math.min(1, Math.max(0, p)), el);
        };
        const onScroll = () => { if (!frame) frame = requestAnimationFrame(update); };
        update();
        window.addEventListener("scroll", onScroll, { passive: true });
        window.addEventListener("resize", onScroll);
        return () => {
            window.removeEventListener("scroll", onScroll);
            window.removeEventListener("resize", onScroll);
            if (frame) cancelAnimationFrame(frame);
        };
    }, [start, end]);
    return ref;
}

/** Grows from slightly smaller to full size as it scrolls up the screen (Apple product-shot zoom). */
export function ScrollZoom({ children, className = "", from = 0.86 }: { children: React.ReactNode; className?: string; from?: number }) {
    const ref = useScrollProgress<HTMLDivElement>((p, el) => {
        const eased = 1 - Math.pow(1 - p, 3);
        el.style.transform = `scale(${from + (1 - from) * eased})`;
        el.style.opacity = String(0.35 + 0.65 * eased);
    }, 1.05, 0.45);
    return <div ref={ref} className={`origin-top will-change-transform ${className}`}>{children}</div>;
}

/** A statement whose words light up one by one as it is scrolled through. */
export function ScrollWords({ text, className = "" }: { text: string; className?: string }) {
    const words = text.split(" ");
    const ref = useScrollProgress<HTMLParagraphElement>((p, el) => {
        const spans = el.querySelectorAll<HTMLSpanElement>("span[data-w]");
        const lit = p * (spans.length + 2);
        spans.forEach((s, i) => { s.style.opacity = String(Math.min(1, Math.max(0.18, lit - i))); });
    }, 0.9, 0.3);
    return (
        <p ref={ref} className={className}>
            {words.map((w, i) => (
                <span key={i} data-w className="transition-opacity duration-200">{w}{i < words.length - 1 ? " " : ""}</span>
            ))}
        </p>
    );
}

/** Drifts up and fades out as the page scrolls away from it (hero copy). */
export function ScrollFade({ children, className = "" }: { children: React.ReactNode; className?: string }) {
    const ref = useScrollProgress<HTMLDivElement>((p, el) => {
        el.style.transform = `translate3d(0, ${-p * 70}px, 0)`;
        el.style.opacity = String(1 - p * 0.85);
    }, 0.1, -0.55);
    return <div ref={ref} className={`will-change-transform ${className}`}>{children}</div>;
}

/** Counts a number such as "0.17 ms", "92.1%" or "+38%" up from zero when it scrolls into view.
 *  The final value is what renders without JavaScript. */
export function CountUp({ value, duration = 1.4 }: { value: string; duration?: number }) {
    const ref = useRef<HTMLSpanElement>(null);
    useEffect(() => {
        const el = ref.current;
        const m = /^([^\d]*)(\d+(?:\.\d+)?)(.*)$/.exec(value);
        if (!el || !m || prefersReducedMotion() || !("IntersectionObserver" in window)) return;
        const target = parseFloat(m[2]);
        const decimals = (m[2].split(".")[1] || "").length;
        const fmt = (x: number) => `${m[1]}${x.toFixed(decimals)}${m[3]}`;
        el.textContent = fmt(0);
        let frame = 0;
        const io = new IntersectionObserver(([entry]) => {
            if (!entry.isIntersecting) return;
            io.disconnect();
            const t0 = performance.now();
            const tick = (t: number) => {
                const k = Math.min(1, (t - t0) / (duration * 1000));
                el.textContent = fmt(target * (1 - Math.pow(1 - k, 3)));
                if (k < 1) frame = requestAnimationFrame(tick);
            };
            frame = requestAnimationFrame(tick);
        }, { threshold: 0.4 });
        io.observe(el);
        return () => { io.disconnect(); cancelAnimationFrame(frame); el.textContent = value; };
    }, [value, duration]);
    return <span ref={ref}>{value}</span>;
}
