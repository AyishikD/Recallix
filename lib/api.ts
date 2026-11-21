// Recallix API client for the dashboard pages.
// Configure at build time (never commit keys):
//   NEXT_PUBLIC_RECALLIX_API_URL  e.g. http://localhost:8000
//   NEXT_PUBLIC_RECALLIX_API_KEY  a *user* key for the local dashboard
//                                 (python -m backend.tools.keys create --user you)
// Note: NEXT_PUBLIC_* values are embedded in the static bundle. Do not set a
// key for a publicly deployed build.
export const API_BASE = process.env.NEXT_PUBLIC_RECALLIX_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_RECALLIX_API_KEY || "";

const headers = (): Record<string, string> => ({
    "Content-Type": "application/json",
    ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
});

async function api<T = any>(method: string, path: string, body?: any): Promise<{ status: string; data: T; timestamp: string }> {
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            method,
            headers: headers(),
            body: body ? JSON.stringify(body) : undefined,
        });
        return res.json();
    } catch (e: any) {
        return { status: "error", data: null as any, timestamp: new Date().toISOString() };
    }
}

export const memoryApi = {
    store: (content: string, agentId = "default_agent", memoryType: "private" | "shared" = "private") =>
        api("POST", "/memory/store", { content, agent_id: agentId, memory_type: memoryType }),
    recall: (query: string, limit = 10, agentId = "default_agent") =>
        api("POST", "/memory/recall", { query, limit, agent_id: agentId }),
    feedback: (recallId: string, relevantMemoryIds: number[]) =>
        api("POST", "/memory/feedback", { recall_id: recallId, relevant_memory_ids: relevantMemoryIds }),
    remove: (memoryId: number) => api("DELETE", `/memory/${memoryId}`),
    timeline: () => api<any[]>("GET", "/memory/timeline"),
    insights: () => api<any[]>("GET", "/memory/insights"),
    worldState: () => api<Record<string, any>>("GET", "/memory/world-state"),
    prediction: () => api<any>("GET", "/memory/prediction"),
    plan: () => api<any[]>("GET", "/memory/plan"),
    stats: () => api<{ episodic_memories: number; timeline_events: number; reflections: number; archived_memories: number; world_state_keys: number }>("GET", "/memory/stats"),
    systemStats: () => api<{ engine: any; user_index: { live: number; pending: number; dim: number; M: number; efSearch: number; max_level: number } | null }>("GET", "/memory/system/stats"),
    profile: () => api<any>("GET", "/memory/profile"),
};

export const agentApi = {
    store: (agentId: string, content: string, memoryType: "private" | "shared" = "private") =>
        api("POST", "/agents/store", { agent_id: agentId, content, memory_type: memoryType }),
    recall: (agentId: string, query: string, limit = 10) =>
        api("POST", "/agents/recall", { agent_id: agentId, query, limit }),
    list: () => api<any[]>("GET", "/agents/list"),
};
