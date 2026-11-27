# recallix (TypeScript)

Client for [Recallix](https://github.com/AyishikD/Recallix): an LLM-free memory layer for AI
agents. Recallix runs on your own machine, needs no LLM calls, and declines instead of guessing.

```bash
npm install recallix
```

You need a running Recallix server. The quickest way is
`docker compose up -d` in the [repository](https://github.com/AyishikD/Recallix); see the quickstart there.

```ts
import { Memory } from "recallix";

const mem = new Memory({ apiKey: process.env.RECALLIX_API_KEY! });   // baseUrl defaults to http://localhost:8000
await mem.store("I moved to Pune last month for a job at Infosys.");
const out = await mem.recall("Where do I live?");
console.log(out.status, out.memories.map((m) => m.content));
await mem.used(out);                                                  // one line of feedback: Recallix learns from it on its own
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

## Changes

* **0.2.6**: `storeMany(items)` stores many messages in one request per 100 (`POST /memory/store_batch`);
  `store(content, { timestamp })` keeps when a message was originally said.
* **0.2.5**: `recall(query, { mode: "cautious" | "balanced" | "helpful" })`. (Version aligned with the Python SDK.)

Node 18+ (uses the built-in `fetch`). License: Apache-2.0.
