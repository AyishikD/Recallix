"use client";

import { useEffect, useRef } from "react";

/* A small cartoon world for the footer: colourful agents wander over night hills, meet,
 * swap memories, and visit the Recallix hub. Pure canvas, no assets or libraries.
 * Pauses off-screen / in background tabs; a single still frame under reduced motion. */

const PALETTE = ["#FF6B5B", "#F4EDE0", "#FFB454", "#5EC8B8", "#8FB3FF", "#FF9A8B", "#F2C14E"];

interface Agent {
    x: number; vx: number; color: string; phase: number;
    state: "walk" | "chat" | "visit"; until: number; face: number; cooldown: number; blink: number;
}
interface Flyer { x0: number; y0: number; x1: number; y1: number; t0: number; dur: number; color: string; toHub: boolean }

export function AgentWorld({ height = 260 }: { height?: number }) {
    const ref = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = ref.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        let seed = 23;
        const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        let W = 0, H = 0;
        const resize = () => {
            const dpr = Math.min(2, window.devicePixelRatio || 1);
            const oldW = W;
            W = canvas.clientWidth; H = canvas.clientHeight;
            canvas.width = W * dpr; canvas.height = H * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            if (oldW) agents.forEach(a => { a.x = (a.x / oldW) * W; });
        };
        const ground = (x: number) => H - 44 + Math.sin(x * 0.006 + 0.6) * 9 + Math.sin(x * 0.017 + 1.3) * 3.5;
        const farHill = (x: number) => H - 78 + Math.sin(x * 0.0042 + 2.1) * 16 + Math.sin(x * 0.011) * 6;

        const stars = Array.from({ length: 70 }, () => ({ x: rnd(), y: rnd() * 0.62, r: 0.4 + rnd() * 1.1, p: rnd() * Math.PI * 2 }));
        const agents: Agent[] = [];
        resize();
        const count = W < 640 ? 5 : PALETTE.length;
        for (let i = 0; i < count; i++) {
            const dir = rnd() > 0.5 ? 1 : -1;
            agents.push({
                x: ((i + 0.5) / count) * W, vx: dir * (0.025 + rnd() * 0.03), color: PALETTE[i], phase: rnd() * 10,
                state: "walk", until: 0, face: dir, cooldown: 1500 + rnd() * 2500, blink: 2000 + rnd() * 4000,
            });
        }
        const flyers: Flyer[] = [];
        const pulses: number[] = [];
        let pointer: { x: number; y: number } | null = null;
        let now = 0;

        const hubX = () => W / 2;
        const hubY = () => ground(hubX()) - 34;

        const send = (x0: number, y0: number, x1: number, y1: number, color: string, toHub: boolean) =>
            flyers.push({ x0, y0, x1, y1, t0: now, dur: 900, color, toHub });

        const step = (dt: number) => {
            for (const a of agents) {
                if (a.state === "chat") {
                    if (now > a.until) { a.state = "walk"; a.vx = (rnd() > 0.5 ? 1 : -1) * (0.025 + rnd() * 0.03); a.face = Math.sign(a.vx); }
                    continue;
                }
                if (a.state === "visit") {
                    const dx = hubX() - a.x;
                    a.vx = Math.sign(dx) * 0.05; a.face = Math.sign(dx) || a.face;
                    if (Math.abs(dx) < 34) {
                        const store = rnd() > 0.45;
                        const ay = ground(a.x) - 42;
                        if (store) send(a.x, ay, hubX(), hubY(), a.color, true);
                        else send(hubX(), hubY(), a.x, ay, a.color, false);
                        pulses.push(now);
                        a.state = "chat"; a.until = now + 900; a.cooldown = now + 4000;
                        continue;
                    }
                } else if (now > a.cooldown && rnd() < dt * 0.00004) {
                    a.state = "visit";
                }
                a.x += a.vx * dt;
                if (a.x < 24) { a.x = 24; a.vx = Math.abs(a.vx); }
                if (a.x > W - 24) { a.x = W - 24; a.vx = -Math.abs(a.vx); }
                if (a.state === "walk") a.face = Math.sign(a.vx) || a.face;
            }
            // meetings
            for (let i = 0; i < agents.length; i++) {
                for (let j = i + 1; j < agents.length; j++) {
                    const a = agents[i], b = agents[j];
                    if (a.state !== "walk" || b.state !== "walk" || now < a.cooldown || now < b.cooldown) continue;
                    if (Math.abs(a.x - b.x) > 50) continue;
                    const until = now + 1900;
                    a.state = b.state = "chat"; a.until = b.until = until;
                    a.cooldown = b.cooldown = now + 5200;
                    a.face = Math.sign(b.x - a.x) || 1; b.face = -a.face;
                    const [from, to] = rnd() > 0.5 ? [a, b] : [b, a];
                    send(from.x, ground(from.x) - 44, to.x, ground(to.x) - 44, from.color, false);
                }
            }
            for (let i = flyers.length - 1; i >= 0; i--) if (now - flyers[i].t0 > flyers[i].dur) flyers.splice(i, 1);
            for (let i = pulses.length - 1; i >= 0; i--) if (now - pulses[i] > 1400) pulses.splice(i, 1);
        };

        const roundRect = (x: number, y: number, w: number, h: number, r: number) => {
            ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
        };

        const SIZE = 1.35;
        const drawAgent = (a: Agent) => {
            const gy = ground(a.x);
            ctx.save(); ctx.translate(a.x, gy); ctx.scale(SIZE, SIZE); ctx.translate(-a.x, -gy);
            const walking = a.state !== "chat";
            const bob = walking ? Math.abs(Math.sin(now * 0.011 + a.phase)) * 5 : Math.sin(now * 0.006 + a.phase) * 1.2;
            const bx = a.x, by = gy - 32 - bob;
            // shadow
            ctx.fillStyle = "rgba(0,0,0,0.35)";
            ctx.beginPath(); ctx.ellipse(bx, gy + 1, 13 - bob * 0.6, 3.2, 0, 0, Math.PI * 2); ctx.fill();
            // legs
            ctx.strokeStyle = a.color; ctx.lineWidth = 3.2; ctx.lineCap = "round";
            const swing = walking ? Math.sin(now * 0.022 + a.phase) * 4 : 0;
            ctx.beginPath();
            ctx.moveTo(bx - 5, by + 26); ctx.lineTo(bx - 5 + swing, gy - 1);
            ctx.moveTo(bx + 5, by + 26); ctx.lineTo(bx + 5 - swing, gy - 1);
            ctx.stroke();
            // antenna
            const glow = a.state === "chat" ? 0.6 + 0.4 * Math.sin(now * 0.02) : 0.25;
            ctx.strokeStyle = a.color; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(bx, by + 1); ctx.quadraticCurveTo(bx + a.face * 3, by - 6, bx + a.face * 2, by - 10); ctx.stroke();
            ctx.fillStyle = `rgba(255,255,255,${glow})`;
            ctx.beginPath(); ctx.arc(bx + a.face * 2, by - 11, 3.6, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = a.color;
            ctx.beginPath(); ctx.arc(bx + a.face * 2, by - 11, 2.4, 0, Math.PI * 2); ctx.fill();
            // body
            ctx.fillStyle = a.color;
            roundRect(bx - 14, by, 28, 28, 11); ctx.fill();
            ctx.fillStyle = "rgba(255,255,255,0.28)";
            ctx.beginPath(); ctx.ellipse(bx - 6, by + 6, 5, 3, -0.5, 0, Math.PI * 2); ctx.fill();
            // eyes — follow the pointer when it is near, else look where walking
            let lx = a.face * 1.6, ly = 0;
            if (pointer) {
                const dx = pointer.x - bx, dy = pointer.y - (by + 11), d = Math.hypot(dx, dy);
                if (d < 260 && d > 0) { lx = (dx / d) * 1.9; ly = (dy / d) * 1.9; }
            }
            const blinking = (now + a.phase * 1000) % a.blink < 130;
            for (const ex of [-5.5, 5.5]) {
                const cx = bx + ex + a.face * 1.5, cy = by + 11;
                if (blinking) {
                    ctx.strokeStyle = "#14213D"; ctx.lineWidth = 1.8;
                    ctx.beginPath(); ctx.moveTo(cx - 3, cy); ctx.lineTo(cx + 3, cy); ctx.stroke();
                } else {
                    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cx, cy, 4.3, 0, Math.PI * 2); ctx.fill();
                    ctx.fillStyle = "#14213D"; ctx.beginPath(); ctx.arc(cx + lx, cy + ly, 2.1, 0, Math.PI * 2); ctx.fill();
                }
            }
            // smile
            ctx.strokeStyle = "rgba(20,33,61,0.75)"; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(bx + a.face * 1.5, by + 17, 3.2, 0.15 * Math.PI, 0.85 * Math.PI); ctx.stroke();
            // typing bubble while chatting
            if (a.state === "chat" && a.until - now > 300) {
                const tx = bx + a.face * 14, ty = by - 22;
                ctx.fillStyle = "#F4EDE0";
                roundRect(tx - 15, ty - 9, 30, 18, 9); ctx.fill();
                ctx.beginPath(); ctx.moveTo(tx - a.face * 6, ty + 8); ctx.lineTo(tx - a.face * 12, ty + 14); ctx.lineTo(tx - a.face * 1, ty + 8); ctx.fill();
                for (let k = 0; k < 3; k++) {
                    const up = Math.max(0, Math.sin(now * 0.012 - k * 0.9)) * 2;
                    ctx.fillStyle = a.color;
                    ctx.beginPath(); ctx.arc(tx - 7 + k * 7, ty - up, 2.1, 0, Math.PI * 2); ctx.fill();
                }
            }
            ctx.restore();
        };

        const draw = () => {
            ctx.clearRect(0, 0, W, H);
            // sky
            const sky = ctx.createLinearGradient(0, 0, 0, H);
            sky.addColorStop(0, "#0E182E"); sky.addColorStop(0.4, "#111D38"); sky.addColorStop(1, "#1A2B52");
            ctx.fillStyle = sky; ctx.fillRect(0, 0, W, H);
            for (const s of stars) {
                ctx.fillStyle = `rgba(255,255,255,${(0.25 + 0.55 * (0.5 + 0.5 * Math.sin(now * 0.0016 + s.p))) * (s.y * 1.6)})`;
                ctx.beginPath(); ctx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2); ctx.fill();
            }
            // hills
            ctx.fillStyle = "#1C2D57";
            ctx.beginPath(); ctx.moveTo(0, H);
            for (let x = 0; x <= W; x += 8) ctx.lineTo(x, farHill(x));
            ctx.lineTo(W, H); ctx.fill();
            ctx.fillStyle = "#24396B";
            ctx.beginPath(); ctx.moveTo(0, H);
            for (let x = 0; x <= W; x += 6) ctx.lineTo(x, ground(x));
            ctx.lineTo(W, H); ctx.fill();
            ctx.strokeStyle = "rgba(255,107,91,0.35)"; ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(0, ground(0));
            for (let x = 6; x <= W; x += 6) ctx.lineTo(x, ground(x));
            ctx.stroke();

            // connections between nearby agents
            ctx.setLineDash([4, 6]); ctx.lineDashOffset = -now / 40; ctx.lineWidth = 1.6;
            for (let i = 0; i < agents.length; i++) {
                for (let j = i + 1; j < agents.length; j++) {
                    const a = agents[i], b = agents[j];
                    const d = Math.abs(a.x - b.x);
                    if (d > 220) continue;
                    const ay = ground(a.x) - 40, byy = ground(b.x) - 40;
                    const g = ctx.createLinearGradient(a.x, ay, b.x, byy);
                    g.addColorStop(0, a.color); g.addColorStop(1, b.color);
                    ctx.strokeStyle = g; ctx.globalAlpha = 0.55 * (1 - d / 220);
                    ctx.beginPath(); ctx.moveTo(a.x, ay); ctx.quadraticCurveTo((a.x + b.x) / 2, Math.min(ay, byy) - 26 - d * 0.08, b.x, byy); ctx.stroke();
                }
            }
            // links from agents walking to the hub
            for (const a of agents) {
                if (a.state !== "visit") continue;
                ctx.strokeStyle = a.color; ctx.globalAlpha = 0.6;
                ctx.beginPath(); ctx.moveTo(a.x, ground(a.x) - 40); ctx.lineTo(hubX(), hubY()); ctx.stroke();
            }
            ctx.setLineDash([]); ctx.globalAlpha = 1;

            // hub
            const hx = hubX(), hy = hubY();
            for (const p of pulses) {
                const k = (now - p) / 1400;
                ctx.strokeStyle = `rgba(255,107,91,${0.7 * (1 - k)})`; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(hx, hy, 18 + k * 46, 0, Math.PI * 2); ctx.stroke();
            }
            const halo = ctx.createRadialGradient(hx, hy, 4, hx, hy, 46);
            halo.addColorStop(0, "rgba(255,107,91,0.42)"); halo.addColorStop(1, "rgba(255,107,91,0)");
            ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(hx, hy, 46, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "#24396B"; ctx.beginPath(); ctx.moveTo(hx - 10, hy + 14); ctx.lineTo(hx + 10, hy + 14); ctx.lineTo(hx + 6, ground(hx)); ctx.lineTo(hx - 6, ground(hx)); ctx.fill();
            const core = ctx.createRadialGradient(hx - 5, hy - 6, 2, hx, hy, 18);
            core.addColorStop(0, "#FFF7EC"); core.addColorStop(1, "#FF6B5B");
            ctx.fillStyle = core; ctx.beginPath(); ctx.arc(hx, hy, 16 + Math.sin(now * 0.003) * 1.2, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "#14213D";
            ctx.beginPath(); ctx.arc(hx, hy, 6.5, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "#F4EDE0";
            ctx.beginPath(); ctx.arc(hx, hy, 3, 0, Math.PI * 2); ctx.fill();
            ctx.font = "600 12px -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif";
            ctx.textAlign = "center"; ctx.fillStyle = "rgba(244,237,224,0.9)";
            ctx.fillText("Recallix", hx, hy - 28);

            // flying memory cards
            for (const f of flyers) {
                const k = Math.min(1, (now - f.t0) / f.dur);
                const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
                const x = f.x0 + (f.x1 - f.x0) * e;
                const y = f.y0 + (f.y1 - f.y0) * e - Math.sin(Math.PI * e) * (f.toHub ? 46 : 34);
                ctx.save(); ctx.translate(x, y); ctx.rotate(Math.sin(k * Math.PI * 2) * 0.25);
                ctx.fillStyle = "#F4EDE0"; roundRect(-8, -6, 16, 12, 3); ctx.fill();
                ctx.fillStyle = f.color; ctx.fillRect(-5, -2.5, 10, 1.8); ctx.fillRect(-5, 1, 6, 1.8);
                ctx.restore();
            }

            agents.slice().sort((a, b) => a.x - b.x).forEach(drawAgent);
        };

        let raf = 0, last = 0, running = false, visible = true;
        const loop = (t: number) => {
            const dt = Math.min(50, last ? t - last : 16);
            last = t; now += dt;
            step(dt); draw();
            raf = requestAnimationFrame(loop);
        };
        const start = () => { if (!running && !reduce && visible && !document.hidden) { running = true; last = 0; raf = requestAnimationFrame(loop); } };
        const stop = () => { running = false; cancelAnimationFrame(raf); };
        const onResize = () => { resize(); if (!running) draw(); };
        const onMove = (e: PointerEvent) => {
            const r = canvas.getBoundingClientRect();
            pointer = e.clientY >= r.top - 80 && e.clientY <= r.bottom + 40 ? { x: e.clientX - r.left, y: e.clientY - r.top } : null;
        };
        const onVisibility = () => (document.hidden ? stop() : start());

        draw();
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
    }, []);

    return (
        <canvas ref={ref} role="img" aria-label="Animated scene: small agents walk over hills, share memories with each other and visit the Recallix hub."
            className="block w-full" style={{ height }} />
    );
}
