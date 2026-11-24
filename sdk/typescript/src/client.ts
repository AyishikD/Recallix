export interface MemoryConfig {
    apiKey: string;
    baseUrl?: string;
}

export class RecallixError extends Error {
    constructor(public status: number, public detail: string) {
        super(`HTTP ${status}: ${detail}`);
    }
}

export class MemoryClient {
    private apiKey: string;
    private baseUrl: string;

    constructor(config: MemoryConfig) {
        if (!config.apiKey) throw new Error("apiKey is required");
        this.apiKey = config.apiKey;
        this.baseUrl = (config.baseUrl || "http://localhost:8000").replace(/\/$/, "");
    }

    /**
     * Returns the response's `data`; throws RecallixError on HTTP errors.
     * Responses default to `any` so callers can read fields (`out.memories[0].content`) without
     * declaring types; pass `T` to type them.
     */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- untyped JSON by default, see above
    async request<T = any>(method: string, endpoint: string, data?: Record<string, unknown>): Promise<T> {
        const body = data ? JSON.stringify(Object.fromEntries(Object.entries(data).filter(([, v]) => v !== undefined))) : undefined;
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method,
            headers: { "X-API-Key": this.apiKey, "Content-Type": "application/json" },
            body,
        });
        const json: { data?: unknown; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok) throw new RecallixError(response.status, json.detail ?? response.statusText);
        return json.data as T;
    }
}
