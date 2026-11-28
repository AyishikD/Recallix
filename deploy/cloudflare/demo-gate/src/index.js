/**
 * The gate in front of the public demo (https://www.recallic.com/demo): demo-api.recallic.com.
 *
 *   POST /session  { turnstile }        -> { token }   one human check (Cloudflare Turnstile) per visit
 *   GET  /health                         (token)      wakes the demo server
 *   POST /store    { content }           (token)      at most 5 messages per visitor per day
 *   POST /recall   { query }             (token)
 *
 * Every limit lives in Upstash Redis, so it survives the demo server sleeping or being redeployed:
 *   messages:  5 per session and 5 per IP per day          requests: 20 per IP per minute
 *   all visitors together: DAILY_REQUESTS per UTC day        sessions: 20 per IP per day
 * Requests reach the demo server (Modal) only from here, with a secret it checks (GATE_SECRET) and the
 * visitor's real IP; the demo server refuses everything else.
 *
 * Secrets (wrangler secret put): TURNSTILE_SECRET, UPSTASH_URL, UPSTASH_TOKEN, GATE_SECRET, SESSION_SECRET.
 */

const ORIGINS = ["https://www.recallic.com", "https://recallic.com"];
const MESSAGES_PER_DAY = 5;
const REQUESTS_PER_MINUTE = 20;
const SESSIONS_PER_IP_PER_DAY = 20;
const DAILY_REQUESTS = 3000;
const SESSION_HOURS = 24;

const enc = new TextEncoder();

function cors(origin) {
    const allowed = ORIGINS.includes(origin) || /^http:\/\/localhost:\d+$/.test(origin || "");
    return allowed ? { "Access-Control-Allow-Origin": origin, "Vary": "Origin",
                       "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                       "Access-Control-Allow-Headers": "Content-Type, Authorization", "Access-Control-Max-Age": "86400" } : {};
}

function reply(status, body, origin) {
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...cors(origin) } });
}

const b64url = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

async function hmac(secret, text) {
    const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    return b64url(await crypto.subtle.sign("HMAC", key, enc.encode(text)));
}

function sameString(a, b) {  // constant time
    if (a.length !== b.length) return false;
    let d = 0;
    for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
    return d === 0;
}

async function newToken(env) {
    const sid = b64url(crypto.getRandomValues(new Uint8Array(16)));
    const exp = Math.floor(Date.now() / 1000) + SESSION_HOURS * 3600;
    return `${sid}.${exp}.${await hmac(env.SESSION_SECRET, `${sid}.${exp}`)}`;
}

async function sessionOf(request, env) {
    const token = (request.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "");
    const [sid, exp, sig] = token.split(".");
    if (!sid || !exp || !sig || Number(exp) < Date.now() / 1000) return null;
    return sameString(sig, await hmac(env.SESSION_SECRET, `${sid}.${exp}`)) ? sid : null;
}

/** Runs Redis commands in one round trip; returns their results. */
async function redis(env, commands) {
    const r = await fetch(`${env.UPSTASH_URL}/pipeline`, {
        method: "POST", headers: { Authorization: `Bearer ${env.UPSTASH_TOKEN}` }, body: JSON.stringify(commands) });
    if (!r.ok) throw new Error(`limits store ${r.status}`);
    return (await r.json()).map((x) => x.result);
}

/** Counts this request against the per-minute and daily limits; returns an error message or null. */
async function countRequest(env, ip) {
    const minute = Math.floor(Date.now() / 60000), day = new Date().toISOString().slice(0, 10);
    const [perMinute, , all] = await redis(env, [
        ["INCR", `rl:m:${ip}:${minute}`], ["EXPIRE", `rl:m:${ip}:${minute}`, "120"],
        ["INCR", `rl:day:${day}`], ["EXPIRE", `rl:day:${day}`, "172800"]]);
    if (all > DAILY_REQUESTS) return "The demo has had a busy day. Please come back tomorrow.";
    if (perMinute > REQUESTS_PER_MINUTE) return "Too many requests from you in a minute; try again shortly.";
    return null;
}

/**
 * Permanent usage totals, one Redis hash per UTC day (stats:YYYY-MM-DD): sessions (passed the human check), chatted
 * (sessions that stored a first message), messages, questions. Unique visitors are a HyperLogLog of a salted hash of
 * the IP (an estimate; the IPs cannot be read back). No message text and no IP is stored. Written after the reply is
 * sent and failures are ignored, so counting never slows or breaks the demo. Read with scripts/demo_stats.py.
 */
function track(ctx, env, ip, fields) {
    const day = new Date().toISOString().slice(0, 10);
    const work = (async () => {
        const cmds = Object.entries(fields).map(([f, n]) => ["HINCRBY", `stats:${day}`, f, String(n)]);
        if (fields.sessions) cmds.push(["PFADD", `stats:visitors:${day}`, await hmac(env.SESSION_SECRET, `visitor:${day}:${ip}`)]);
        await redis(env, cmds);
    })().catch(() => {});
    if (ctx && ctx.waitUntil) ctx.waitUntil(work);
}

async function forward(env, path, init, ip) {
    const headers = { "Content-Type": "application/json", "X-Demo-Gate": env.GATE_SECRET, "X-Demo-Client": ip };
    const r = await fetch(`${env.DEMO_ORIGIN}${path}`, { ...init, headers });
    return { status: r.status, body: await r.json().catch(() => ({ detail: "The demo server answered oddly." })) };
}

export default {
    async fetch(request, env, ctx) {
        const url = new URL(request.url), origin = request.headers.get("Origin") || "";
        if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
        const ip = request.headers.get("CF-Connecting-IP") || "unknown";
        try {
            if (url.pathname === "/session" && request.method === "POST") {
                const { turnstile } = await request.json().catch(() => ({}));
                if (!turnstile || typeof turnstile !== "string") return reply(400, { detail: "Missing the human check." }, origin);
                const form = new FormData();
                form.append("secret", env.TURNSTILE_SECRET); form.append("response", turnstile); form.append("remoteip", ip);
                const check = await (await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", { method: "POST", body: form })).json();
                if (!check.success) return reply(403, { detail: "The human check did not pass. Please reload the page." }, origin);
                const day = new Date().toISOString().slice(0, 10);
                const [sessions] = await redis(env, [["INCR", `rl:s:${ip}:${day}`], ["EXPIRE", `rl:s:${ip}:${day}`, "172800"]]);
                if (sessions > SESSIONS_PER_IP_PER_DAY) return reply(429, { detail: "Too many new visits from you today; come back tomorrow." }, origin);
                track(ctx, env, ip, { sessions: 1 });
                return reply(200, { token: await newToken(env) }, origin);
            }

            const sid = await sessionOf(request, env);
            if (!sid) return reply(401, { detail: "Your demo session expired. Please reload the page." }, origin);
            const limited = await countRequest(env, ip);
            if (limited) return reply(429, { detail: limited }, origin);

            if (url.pathname === "/health" && request.method === "GET") {
                const r = await forward(env, "/health", { method: "GET" }, ip);
                return reply(r.status, r.body, origin);
            }
            if (url.pathname === "/store" && request.method === "POST") {
                const { content } = await request.json().catch(() => ({}));
                if (typeof content !== "string" || !content.trim() || content.length > 400)
                    return reply(422, { detail: "A message is 1 to 400 characters." }, origin);
                const day = new Date().toISOString().slice(0, 10);
                const [bySession, , byIp] = await redis(env, [
                    ["INCR", `rl:msg:s:${sid}`], ["EXPIRE", `rl:msg:s:${sid}`, "172800"],
                    ["INCR", `rl:msg:ip:${ip}:${day}`], ["EXPIRE", `rl:msg:ip:${ip}:${day}`, "172800"]]);
                if (bySession > MESSAGES_PER_DAY || byIp > MESSAGES_PER_DAY)
                    return reply(429, { detail: `The demo keeps ${MESSAGES_PER_DAY} messages per visitor. Ask about them, or come back tomorrow for a fresh start.` }, origin);
                const r = await forward(env, "/store", { method: "POST", body: JSON.stringify({ session: sid, content }) }, ip);
                if (r.status < 300) track(ctx, env, ip, bySession === 1 ? { messages: 1, chatted: 1 } : { messages: 1 });
                return reply(r.status, r.body, origin);
            }
            if (url.pathname === "/recall" && request.method === "POST") {
                const { query } = await request.json().catch(() => ({}));
                if (typeof query !== "string" || !query.trim() || query.length > 400)
                    return reply(422, { detail: "A question is 1 to 400 characters." }, origin);
                const r = await forward(env, "/recall", { method: "POST", body: JSON.stringify({ session: sid, query }) }, ip);
                if (r.status < 300) track(ctx, env, ip, { questions: 1 });
                return reply(r.status, r.body, origin);
            }
            return reply(404, { detail: "Not found." }, origin);
        } catch (e) {
            return reply(503, { detail: "The demo is unavailable right now. Please try again in a minute." }, origin);
        }
    },
};
