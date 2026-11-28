# recallic (TypeScript)

Client for [Recallic](https://github.com/Recallic/Recallic): an LLM-free memory layer for AI
agents. Recallic runs on your own machine, needs no LLM calls, and declines instead of guessing.

```bash
npm install recallic
```

You need a running Recallic server. The quickest way is
`docker compose up -d` in the [repository](https://github.com/Recallic/Recallic); see the quickstart there.

```ts
import { Memory } from "recallic";

const mem = new Memory({ apiKey: process.env.RECALLIC_API_KEY! });   // baseUrl defaults to http://localhost:8000
await mem.store("I moved to Pune last month for a job at Infosys.");
const out = await mem.recall("Where do I live?");
console.log(out.status, out.memories.map((m) => m.content));
await mem.used(out);                                                  // one line of feedback: Recallic learns from it on its own
```

How readily recall answers is one option (server 0.2.5 or newer; default `"balanced"`):

```ts
await mem.recall("Which came first, the Goa trip or the Pune workshop?", { mode: "helpful" });  // fewer refusals, computed_answer
await mem.recall("What dose did the doctor prescribe?", { mode: "cautious" });                    // checks every answer twice
```

Storing a whole conversation at once (server 0.2.6 or newer; an older server gets one store per message):

```ts
await mem.storeMany(["I moved to Pune in March.", { content: "Mira got married.", timestamp: 1717200000 }]);
```

Correcting a memory keeps when it was said; the memory gets a new id (server 0.2.7 or newer):

```ts
await mem.update(memoryId, "I moved to Mumbai in March.");
```

## Changes

* **0.3.0**: renamed from `recallix` to **`recallic`**: `npm install recallic`, `import { Memory } from "recallic"`.
  Nothing else changed.
* **0.2.7**: `update(memoryId, content)` corrects a memory (`PUT /memory/{id}`).
* **0.2.6**: `storeMany(items)` stores many messages in one request per 100 (`POST /memory/store_batch`);
  `store(content, { timestamp })` keeps when a message was originally said.
* **0.2.5**: `recall(query, { mode: "cautious" | "balanced" | "helpful" })`. (Version aligned with the Python SDK.)

Node 18+ (uses the built-in `fetch`). License: Apache-2.0.
