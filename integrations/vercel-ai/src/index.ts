/**
 * Recallic memory for the Vercel AI SDK.
 *
 *     import { generateText, wrapLanguageModel } from "ai";
 *     import { recallicMemory } from "recallic-ai-sdk";
 *
 *     const model = wrapLanguageModel({ model: openai("gpt-4.1-mini"), middleware: recallicMemory({ apiKey }) });
 *     await generateText({ model, prompt: "Where did I move?", providerOptions: { recallic: { userId: "alice" } } });
 *
 * Before each model call, recalls what this user said before that bears on their latest message and gives it to
 * the model as a system message; when nothing answers it, tells the model there is no record instead of handing
 * over near-misses. Then stores the user's message. No LLM is called to store or search memory.
 */
import { Memory } from "recallic";

export type RecallMode = "cautious" | "balanced" | "helpful";

export interface RecallicMemoryOptions {
    /** A Recallic key. Role "service" acts for each userId; a user's own key acts for itself. */
    apiKey: string;
    /** Default http://localhost:8000. */
    baseUrl?: string;
    /** Whose memory: fixed here, or per call with providerOptions: { recallic: { userId } }. */
    userId?: string;
    /** Recallic agent id; each agent reads back what it wrote (and what any agent shared). Default "ai-sdk". */
    agentId?: string;
    /** Default: the server's recall.mode. */
    mode?: RecallMode;
    /** Memories given to the model. Default 5. */
    limit?: number;
    /** Store the user's latest message after recalling. Default true. */
    storeUserMessages?: boolean;
    /** When nothing answers, tell the model not to guess about the user's past. Default true. */
    tellModelOnDecline?: boolean;
    /**
     * The AI SDK middleware version: "v4" for AI SDK 7 (default), "v3" for AI SDK 6.
     */
    specificationVersion?: "v3" | "v4";
    /** Called when Recallic cannot be reached; the call goes on without memory. Default: console.warn. */
    onError?: (error: unknown) => void;
    /** For tests: a ready client. */
    client?: Pick<Memory, "recall" | "store">;
}

/** Per-call settings: providerOptions: { recallic: { ... } }. */
export interface RecallicCallOptions {
    userId?: string;
    agentId?: string;
    mode?: RecallMode;
    /** Skip memory for this call. */
    disabled?: boolean;
}

export const MARKER = "### Recallic memory";
export const DECLINE_NOTE =
    "None of this user's earlier messages answers their latest message. If it depends on something they would " +
    "have told you before, say you don't have a record of it rather than guess.";

type Part = { type: string; text?: string };
type Message = { role: string; content: string | Part[] };
type Params = { prompt: Message[]; providerOptions?: Record<string, Record<string, unknown> | undefined> };

const CACHE_MS = 60_000;

function messageText(message: Message): string {
    if (typeof message.content === "string") return message.content.trim();
    return message.content.filter((p) => p.type === "text" && p.text).map((p) => p.text).join(" ").trim();
}

function when(ts: unknown): string {
    const n = Number(ts);
    if (!Number.isFinite(n) || n <= 0) return "";
    return new Date(n * 1000).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
}

export function memoryNote(out: any, limit: number, tellModelOnDecline: boolean): string | null {
    const memories: any[] = out?.memories ?? [];
    if (out?.status === "CONSISTENT" && memories.length) {
        const lines = [MARKER, "Things this user told you before (newest facts win; use only what bears on their message):"];
        for (const m of memories.slice(0, limit)) {
            const d = when(m.timestamp);
            lines.push(`- ${d ? `[${d}] ` : ""}${m.content_resolved ?? m.content ?? ""}`);
        }
        const computed = out.computed_answer;
        if (computed?.text) lines.push(`Worked out from these memories: ${computed.text} (${out.receipt?.computed ?? ""}).`);
        return lines.join("\n");
    }
    return tellModelOnDecline ? `${MARKER}\n${DECLINE_NOTE}` : null;
}

/** AI SDK middleware that gives the model the user's Recallic memory. Use with wrapLanguageModel. */
export function recallicMemory(options: RecallicMemoryOptions) {
    const client = options.client ?? new Memory({ apiKey: options.apiKey, baseUrl: options.baseUrl ?? "http://localhost:8000" });
    const limit = options.limit ?? 5;
    const onError = options.onError ?? ((e: unknown) => console.warn("Recallic memory unavailable:", e));
    const cache = new Map<string, { note: string | null; at: number }>();

    async function transformParams({ params }: { type: "generate" | "stream"; params: Params; model?: unknown }) {
        const call = (params.providerOptions?.recallic ?? {}) as RecallicCallOptions;
        if (call.disabled) return params;
        const userId = call.userId ?? options.userId;
        const agentId = call.agentId ?? options.agentId ?? "ai-sdk";
        const mode = call.mode ?? options.mode;
        const prompt = params.prompt ?? [];
        const lastUserIndex = prompt.map((m) => m.role).lastIndexOf("user");
        const text = lastUserIndex >= 0 ? messageText(prompt[lastUserIndex]) : "";
        if (!text) return params;

        // a tool loop calls the model again for the same question: recall once
        const key = JSON.stringify([userId ?? "", agentId, mode ?? "", text]);
        let note: string | null | undefined;
        const hit = cache.get(key);
        if (hit && Date.now() - hit.at < CACHE_MS) {
            note = hit.note;
        } else {
            try {
                const out = await client.recall(text.slice(0, 2000), { limit, agentId, userId, mode });
                note = memoryNote(out, limit, options.tellModelOnDecline ?? true);
                cache.set(key, { note, at: Date.now() });
                if (cache.size > 500) cache.delete(cache.keys().next().value as string);
            } catch (e) {
                onError(e);
                note = undefined;
            }
            // store after recalling, so the question does not recall itself; only on the user's turn
            if (note !== undefined && (options.storeUserMessages ?? true) && lastUserIndex === prompt.length - 1) {
                try {
                    await client.store(text.slice(0, 20000), { agentId, userId, metadata: { speaker: "User" } });
                } catch (e) {
                    onError(e);
                }
            }
        }
        if (!note) return params;

        const withoutOld = prompt.filter((m) => !(m.role === "system" && typeof m.content === "string" && m.content.startsWith(MARKER)));
        let at = 0;
        while (at < withoutOld.length && withoutOld[at].role === "system") at++;
        return { ...params, prompt: [...withoutOld.slice(0, at), { role: "system", content: note }, ...withoutOld.slice(at)] };
    }

    return { specificationVersion: options.specificationVersion ?? "v4", transformParams } as any;
}

export default recallicMemory;
