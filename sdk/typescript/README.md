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

Node 18+ (uses the built-in `fetch`). License: Apache-2.0.
