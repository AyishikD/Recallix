# recallic-ai-sdk: Recallic memory for the Vercel AI SDK

*Formerly `recallix-ai-sdk` (0.1.x); renamed with the project, nothing else changed.*

`recallicMemory()` is AI SDK middleware that gives any model a self-hosted
[Recallic](https://github.com/Recallic/Recallic) memory. Wrap the model once and use it as usual.

* **No LLM calls for memory.** Storing and searching use Recallic's local models.
* **Before each call**, it recalls what this user said before that bears on their latest message and gives it to
  the model as a system message (after your own system prompt).
* **Honest.** When nothing answers it, the model is told there is no record instead of being handed near-misses to
  guess from.
* **Then it stores** the user's message (re-sending one Recallic already holds is skipped).
* **Your data stays on your server**, separated per user and per agent.

## Install

```bash
npm install recallic-ai-sdk ai
```

Run Recallic ([quickstart](../../docs/quickstart.md)) and create a key with role `service`, which acts for each of
your users:

```bash
python -m backend.tools.keys create --user my-ai-app --role service
```

## Use

```ts
import { generateText, wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { recallicMemory } from "recallic-ai-sdk";

const model = wrapLanguageModel({
    model: openai("gpt-4.1-mini"),
    middleware: recallicMemory({ apiKey: process.env.RECALLIC_SERVICE_KEY!, baseUrl: "http://localhost:8000" }),
});

// one wrapped model serves every user: say whose memory per call
const { text } = await generateText({
    model,
    prompt: "Which city did I move to?",
    providerOptions: { recallic: { userId: "alice" } },
});
```

It works the same with `streamText`, tools and multi-step calls (a tool loop recalls once and stores once).

| Option | Default | What it does |
|---|---|---|
| `apiKey` | (required) | A Recallic key. Role `service` acts for each `userId`; a user's own key acts for itself. |
| `baseUrl` | `http://localhost:8000` | The Recallic API. |
| `userId` | none | Whose memory, if fixed; otherwise per call with `providerOptions.recallic.userId`. |
| `agentId` | `ai-sdk` | Recallic agent id; each agent reads back what it wrote (and what any agent shared). |
| `mode` | server's `recall.mode` | `cautious` checks every answer twice; `helpful` answers more counting, ordering and "how long between" questions and adds a worked-out result. |
| `limit` | 5 | Memories given to the model. |
| `storeUserMessages` | `true` | Store the user's latest message after recalling. |
| `tellModelOnDecline` | `true` | When nothing answers, tell the model not to guess about the user's past. |
| `specificationVersion` | `"v4"` | `"v4"` for AI SDK 7, `"v3"` for AI SDK 6. |
| `onError` | `console.warn` | Called when Recallic cannot be reached; the call goes on without memory. |

Per call, `providerOptions.recallic` takes `userId`, `agentId`, `mode`, and `disabled: true` to skip memory.

Needs Recallic server 0.2.5 or newer. Tested with AI SDK 7 and 6, and with a real model through
`@ai-sdk/openai-compatible`.

## Tests

`npm install && npm test`. With a running Recallic, `RECALLIC_URL=... RECALLIC_SERVICE_KEY=rx_... npm test`
also runs a live test.

Licence: Apache-2.0.
