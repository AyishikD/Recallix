import { MemoryClient, MemoryConfig } from "./client";

export type MemoryType = "private" | "shared";

export interface StoreOptions {
    agentId?: string;
    memoryType?: MemoryType;
    /** Only honoured for service/admin keys. */
    userId?: string;
    metadata?: Record<string, any>;
}

export interface RecallOptions {
    limit?: number;
    agentId?: string;
    /** Only honoured for service/admin keys. */
    userId?: string;
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
        });
    }

    /** Mark which recalled memories were useful (trains the ranker). */
    feedback(recallId: string, relevantMemoryIds: number[], userId?: string) {
        return this.client.request("POST", "/memory/feedback", {
            recall_id: recallId, relevant_memory_ids: relevantMemoryIds, user_id: userId,
        });
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
