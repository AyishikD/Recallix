"use client";

import { useState } from "react";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { ButtonLink, Eyebrow, Headline, Lede, Reveal, ScrollFade, Section } from "@/components/site/ui";
import { ClosingCTA, MemoryField, TypeCode } from "@/components/site/story";

/* Code samples mirror sdk/python/memoryos, sdk/typescript and docs/quickstart.md. */

const SAMPLES: Record<string, { label: string; code: string }> = {
    python: {
        label: "Python",
        code: `import os
from memoryos import Memory

mem = Memory(api_key=os.environ["RECALLIX_API_KEY"])

mem.store("My name is Priya and I live in Pune.")
out = mem.recall("Where do I live?")

print(out["status"])                       # CONSISTENT, ABSTAINED or NO_MATCH
for m in out["memories"]:
    print(m["content"], m.get("answer"), m["explain"])

# Tell Recallix which results helped — this trains the ranker.
mem.feedback(out["recall_id"], [out["memories"][0]["id"]])`,
    },
    typescript: {
        label: "TypeScript",
        code: `import { Memory } from "memoryos-sdk";

const mem = new Memory({ apiKey: process.env.RECALLIX_API_KEY! });

await mem.store("My name is Priya and I live in Pune.");
const out = await mem.recall("Where do I live?");

console.log(out.status);
for (const m of out.memories) console.log(m.content, m.answer);

await mem.feedback(out.recall_id, [out.memories[0].id]);`,
    },
    curl: {
        label: "REST",
        code: `curl -X POST http://localhost:8000/memory/store \\
  -H "X-API-Key: $RECALLIX_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "I prefer dark mode.", "agent_id": "settings-bot", "memory_type": "shared"}'

curl -X POST http://localhost:8000/memory/recall \\
  -H "X-API-Key: $RECALLIX_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Which theme does the user like?", "agent_id": "chat-bot"}'`,
    },
};

const multiUser = `# A service key may act for its end users.
app = Memory(api_key=os.environ["RECALLIX_SERVICE_KEY"])

app.store(
    "Prefers vegetarian food.",
    user_id="customer-42",
    agent_id="concierge",
    memory_type="shared",
)
app.recall(
    "Any dietary preferences?",
    user_id="customer-42",
    agent_id="booking",
)`;

const SETUP = [
    { t: "Install", c: `python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-local.txt   # optional: offline embeddings + answer check
cmake -S backend/infra_cpp -B backend/infra_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/infra_cpp/build -j` },
    { t: "Configure", c: `python -m backend.tools.setup_env      # writes .env, prints an admin key once` },
    { t: "Run", c: `set -a; source .env; set +a
./backend/infra_cpp/build/memory_service &   # vector engine
python server.py                             # API on :8000` },
    { t: "Create a key", c: `python -m backend.tools.keys create --user alice
python -m backend.tools.keys create --user my-app --role service` },
];

const ENDPOINTS = [
    ["POST", "/memory/store", "Store a message. Facts, topic and importance are computed for you."],
    ["POST", "/memory/recall", "Ranked memories with status, explanations and the answer span."],
    ["POST", "/memory/feedback", "Mark useful results; used to refit the ranker."],
    ["DELETE", "/memory/{id}", "Removes the memory, its vector and its timeline entry."],
    ["GET", "/memory/timeline", "What was stored, in order."],
    ["GET", "/memory/world-state", "Latest known facts about the user."],
    ["GET", "/memory/insights", "Reflections written by the background job."],
    ["GET", "/memory/stats", "Counts for the dashboard."],
];

export default function DevelopersPage() {
    const [lang, setLang] = useState<keyof typeof SAMPLES>("python");
    return (
        <main>
            <Nav />
            <Section hero className="text-center" backdrop={<MemoryField />}>
                <ScrollFade>
                    <Reveal from="fade"><Eyebrow>Developers</Eyebrow></Reveal>
                    <Headline as="h1">
                        <Reveal as="span" delay={0.08}>Store. Recall.</Reveal>
                        <Reveal as="span" delay={0.2} className="text-sheen">That’s the API.</Reveal>
                    </Headline>
                    <Reveal delay={0.34}>
                        <Lede className="mx-auto mt-6">Python and TypeScript SDKs over a small REST API. Self-hosted, one config file, no LLM to provision.</Lede>
                    </Reveal>
                </ScrollFade>
            </Section>

            <Section tone="cloud">
                <Reveal>
                    <div className="flex justify-center mb-8">
                        <div className="inline-flex rounded-full card-edge p-1">
                            {Object.entries(SAMPLES).map(([k, s]) => (
                                <button key={k} onClick={() => setLang(k as keyof typeof SAMPLES)}
                                    className={`px-5 h-9 rounded-full text-[14px] transition-colors ${lang === k ? "bg-cta text-cta-ink font-medium" : "text-ink-2 hover:text-ink"}`}>
                                    {s.label}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="max-w-[760px] mx-auto"><TypeCode key={lang} code={SAMPLES[lang].code} label={SAMPLES[lang].label} /></div>
                    <p className="text-center mt-6 text-[14px] text-ink-2 max-w-[620px] mx-auto">
                        <strong className="text-ink">status</strong> is CONSISTENT when a memory answers, ABSTAINED when matching memories exist
                        but none answers the question, and NO_MATCH when nothing matches.
                    </p>
                </Reveal>
            </Section>

            <Section>
                <div className="grid gap-12 md:grid-cols-2 md:items-center">
                    <Reveal from="left">
                        <Eyebrow>Many users, many agents</Eyebrow>
                        <Headline>Built for products, not demos.</Headline>
                        <Lede className="mt-5">
                            A user key always acts as its own user. A service key can act for any of your end users — each gets a separate
                            index. Mark a memory <code>shared</code> to make it visible to all of that user’s agents, or keep it <code>private</code> to one.
                        </Lede>
                    </Reveal>
                    <Reveal from="right" delay={0.12}><TypeCode code={multiUser} label="Python · service key" /></Reveal>
                </div>
            </Section>

            <Section tone="cloud">
                <Reveal>
                    <Eyebrow>Set up</Eyebrow>
                    <Headline className="mb-12">Running in four steps.</Headline>
                </Reveal>
                <div className="grid gap-5">
                    {SETUP.map((s, i) => (
                        <Reveal key={s.t} from={i % 2 === 0 ? "left" : "right"} delay={0.05}>
                            <div className="rounded-[28px] card-edge p-7 h-full">
                                <p className="text-[13px] text-ink-3 mb-1">Step {i + 1}</p>
                                <h3 className="type-title mb-4">{s.t}</h3>
                                <pre className="tone-navy text-[12.5px] bg-code rounded-[14px] p-4 whitespace-pre-wrap break-words">{s.c}</pre>
                            </div>
                        </Reveal>
                    ))}
                </div>
                <p className="mt-6 text-[14px] text-ink-2">
                    Embeddings come from Ollama, sentence-transformers in-process, or any OpenAI-compatible endpoint —
                    set <code>models.embedding_provider</code> in config.yaml.
                </p>
            </Section>

            <Section>
                <Reveal>
                    <Eyebrow>Reference</Eyebrow>
                    <Headline className="mb-10">Endpoints.</Headline>
                </Reveal>
                <div className="rounded-[28px] card-edge divide-y divide-line/60 overflow-hidden">
                    {ENDPOINTS.map(([m, p, d], i) => (
                        <Reveal key={p + m} from={i % 2 === 0 ? "left" : "right"} delay={0.03}><div className="grid md:grid-cols-[80px_220px_1fr] gap-1 md:gap-4 px-6 md:px-8 py-4 items-baseline">
                            <span className="text-[12px] font-semibold text-accent">{m}</span>
                            <code className="text-[14px]">{p}</code>
                            <span className="text-ink-2 text-[15px]">{d}</span>
                        </div></Reveal>
                    ))}
                </div>
                <div className="mt-8 flex flex-wrap gap-6">
                    <ButtonLink href="https://github.com/AyishikD/Recallix/blob/main/docs/api_reference.md" variant="link">Full API reference</ButtonLink>
                    <ButtonLink href="https://github.com/AyishikD/Recallix/blob/main/docs/quickstart.md" variant="link">Quickstart</ButtonLink>
                </div>
            </Section>

            <ClosingCTA />

            <Footer />
        </main>
    );
}
