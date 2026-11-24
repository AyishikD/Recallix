// Recallix API client for the dashboard pages.
// Configure at build time (never commit keys):
//   NEXT_PUBLIC_RECALLIX_API_URL  e.g. http://localhost:8000
//   NEXT_PUBLIC_RECALLIX_API_KEY  a *user* key for the local dashboard
//                                 (python -m backend.tools.keys create --user you)
// Note: NEXT_PUBLIC_* values are embedded in the static bundle. Do not set a
// key for a publicly deployed build.
export const API_BASE = process.env.NEXT_PUBLIC_RECALLIX_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_RECALLIX_API_KEY || "";

/** The API's envelope; `data` is null when the server could not be reached. */
export interface Envelope<T> { status: string; data: T | null; timestamp: string }

export interface MemoryStats {
    episodic_memories: number; timeline_events: number; reflections: number;
    archived_memories: number; world_state_keys: number;
}
export interface EngineInfo { status?: string; simd?: string; namespaces?: number }
export interface IndexInfo { live: number; pending: number; dim: number; M: number; efSearch: number; max_level: number }
export interface SystemStats { engine: EngineInfo | null; user_index: IndexInfo | null }
export interface Insight { summary: string; confidence?: number }

const headers = (): Record<string, string> => ({
    "Content-Type": "application/json",
    ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
});

async function api<T>(method: string, path: string, body?: unknown): Promise<Envelope<T>> {
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            method,
            headers: headers(),
            body: body ? JSON.stringify(body) : undefined,
        });
        return (await res.json()) as Envelope<T>;
    } catch {
        return { status: "error", data: null, timestamp: new Date().toISOString() };
    }
}

export const memoryApi = {
    stats: () => api<MemoryStats>("GET", "/memory/stats"),
    systemStats: () => api<SystemStats>("GET", "/memory/system/stats"),
    insights: () => api<Insight[]>("GET", "/memory/insights"),
};
