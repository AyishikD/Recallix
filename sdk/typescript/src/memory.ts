import { MemoryClient, MemoryConfig } from "./client";

export type MemoryType = "private" | "shared";

export interface StoreOptions {
    agentId?: string;
    memoryType?: MemoryType;
    /** Only honoured for service/admin keys. */
    userId?: string;
    metadata?: Record<string, unknown>;
}

export interface RecallOptions {
    limit?: number;
    agentId?: string;
    /** Only honoured for service/admin keys. */
    userId?: string;
    /** Re-read the candidates with a re-ranker: better when the answer is spread over several memories, slower. */
    deep?: boolean;
    /**
     * When recall declines (status "ABSTAINED"), still return the results it found, as low-confidence
     * (`answer` stays null). Check `status` before using them as an answer. Default: server setting (off).
     */
    keepOnDecline?: boolean;
    /**
     * How readily recall answers: "cautious" verifies every answer (more "I don't know", fewer wrong answers),
     * "balanced" is the default, "helpful" answers more counting / ordering / time questions (fewer refusals).
     * Default: server setting (recall.mode).
     */
    mode?: "cautious" | "balanced" | "helpful";
}

const q = (params: Record<string, string | number | undefined>) => {
    const entries = Object.entries(params).filter(([, v]) => v !== undefined) as [string, string][];
    return entries.length ? `?${new URLSearchParams(entries.map(([k, v]) => [k, String(v)]))}` : "";
};

export class Memory {
    private client: MemoryClient;

    constructor(config: MemoryConfig) {
        this.client = new MemoryClient(config);
    }

    store(content: string, opts: StoreOptions = {}) {
        return this.client.request("POST", "/memory/store", {
            content, agent_id: opts.agentId ?? "default_agent", memory_type: opts.memoryType ?? "private",
            user_id: opts.userId, metadata: opts.metadata,
        });
    }

    recall(query: string, opts: RecallOptions = {}) {
        return this.client.request("POST", "/memory/recall", {
            query, limit: opts.limit ?? 10, agent_id: opts.agentId ?? "default_agent", user_id: opts.userId,
            deep: opts.deep, keep_on_decline: opts.keepOnDecline, mode: opts.mode,
        });
    }

    /** Mark which recalled memories were useful (trains the ranker). */
    feedback(recallId: string, relevantMemoryIds: number[], userId?: string) {
        return this.client.request("POST", "/memory/feedback", {
            recall_id: recallId, relevant_memory_ids: relevantMemoryIds, user_id: userId,
        });
    }

    /**
     * One-line feedback after a recall, so Recallix learns from use:
     *
     *     const out = await mem.recall("Where do I live?");
     *     await mem.used(out);                    // the first memory answered it
     *     await mem.used(out, out.memories[2]);   // or name the memories (or ids) your app actually used
     *
     * Recallix refits its ranking from this on a schedule and keeps a new fit only if it is better.
     */
    used(recall: { recall_id: string; memories?: { id: number }[] }, ...memories: ({ id: number } | number)[]) {
        const chosen = memories.length ? memories : (recall.memories ?? []).slice(0, 1);
        return this.feedback(recall.recall_id, chosen.map(m => (typeof m === "number" ? m : m.id)));
    }

    /** Feedback that nothing in this recall was useful. */
    noneUsed(recall: { recall_id: string }) {
        return this.feedback(recall.recall_id, []);
    }

    /** Memories worth keeping in mind for a new message (no question needed), each with the reason. */
    proactive(message: string, opts: { agentId?: string; limit?: number; userId?: string } = {}) {
        return this.client.request("POST", "/memory/proactive", {
            message, agent_id: opts.agentId ?? "default_agent", limit: opts.limit ?? 5, user_id: opts.userId,
        });
    }

    /** All memories in the open "recallix-memory" format. */
    export(userId?: string) {
        return this.client.request("GET", `/memory/export${q({ user_id: userId })}`);
    }

    /** Import a Recallix export, a Mem0 export (format "mem0") or text, one memory per line (format "text"). */
    importMemories(data: unknown, opts: { format?: "recallix" | "mem0" | "text"; agentId?: string; memoryType?: MemoryType; userId?: string } = {}) {
        return this.client.request("POST", "/memory/import", {
            format: opts.format ?? "recallix", data, agent_id: opts.agentId ?? "default_agent",
            memory_type: opts.memoryType ?? "private", user_id: opts.userId,
        });
    }

    delete(memoryId: number, userId?: string) {
        return this.client.request("DELETE", `/memory/${memoryId}${q({ user_id: userId })}`);
    }

    timeline(limit = 50, userId?: string) {
        return this.client.request("GET", `/memory/timeline${q({ limit, user_id: userId })}`);
    }

    insights(userId?: string) {
        return this.client.request("GET", `/memory/insights${q({ user_id: userId })}`);
    }

    worldState(userId?: string) {
        return this.client.request("GET", `/memory/world-state${q({ user_id: userId })}`);
    }

    prediction(userId?: string) {
        return this.client.request("GET", `/memory/prediction${q({ user_id: userId })}`);
    }

    plan(userId?: string) {
        return this.client.request("GET", `/memory/plan${q({ user_id: userId })}`);
    }

    profile(userId?: string) {
        return this.client.request("GET", `/memory/profile${q({ user_id: userId })}`);
    }
}
