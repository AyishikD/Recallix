"use client";

import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { CountUp, Eyebrow, Headline, Lede, Reveal, ScrollFade, Section } from "@/components/site/ui";
import { ClosingCTA, MemoryField } from "@/components/site/story";

type Status = "shipped" | "known gap" | "next" | "exploring";

const STATUS_STYLE: Record<Status, string> = {
    shipped: "bg-good/10 text-good",
    "known gap": "bg-warn/10 text-warn",
    next: "bg-accent/10 text-accent",
    exploring: "bg-cloud text-ink-2",
};

const GROUPS: { title: string; lede: string; items: { name: string; detail: string; status: Status }[] }[] = [
    {
        title: "Shipped",
        lede: "Done, tested, and measured on the benchmarks page.",
        items: [
            { name: "No LLM in the loop", detail: "Facts, topics, importance and ranking are rules and small fitted models.", status: "shipped" },
            { name: "Learned ranking and importance", detail: "Logistic models fitted with cross-validation, refit from your own feedback.", status: "shipped" },
            { name: "Answer check", detail: "A local reading-comprehension model declines when no memory answers the question.", status: "shipped" },
            { name: "Facts that stay current", detail: "Newer values supersede older ones; a denial removes only the value denied.", status: "shipped" },
            { name: "Private and shared memories", detail: "Per-agent visibility enforced in SQL, plus a separate vector index for every user.", status: "shipped" },
            { name: "Rebuild from SQLite", detail: "The vector engine can restart at any time; indexes are restored from the system of record.", status: "shipped" },
            { name: "Decay and reinforcement", detail: "Recalled memories are reinforced; old, unused, unimportant ones are forgotten.", status: "shipped" },
            { name: "Receipts on every recall", detail: "Why it answered or declined, the versions involved, and for each fact the rule and exact quote it came from.", status: "shipped" },
            { name: "Memory tests in CI", detail: "Assert what an agent must remember, and must not invent, in YAML. Deterministic, so a failure means behaviour changed.", status: "shipped" },
            { name: "Replay and diff", detail: "Rebuild memory from the original messages and see exactly what a rule or model change would alter.", status: "shipped" },
            { name: "Calibrated refusals", detail: "Calibrate the decline decision on a deployment’s own questions and get a certificate bounding wrong refusals.", status: "shipped" },
            { name: "Fact half-lives", detail: "Each fact says how likely it is to still be true, learned from how often that kind of fact changed for this user.", status: "shipped" },
            { name: "Confirmed answers", detail: "When feedback confirms an answer, similar later questions always consider that memory.", status: "shipped" },
            { name: "Search that scales with tenants", detail: "Full-text search is restricted to one user inside the index, so recall time no longer grows with other users’ data.", status: "shipped" },
            { name: "Export, import and MCP", detail: "An open export format, imports from Mem0 or plain text, and an MCP server for Claude Desktop, Cursor and other clients.", status: "shipped" },
            { name: "Open benchmark harness", detail: "Any memory system behind a small adapter, scored on the same data in the same way.", status: "shipped" },
        ],
    },
    {
        title: "Known gaps",
        lede: "What does not work well yet. We would rather tell you than have you find out.",
        items: [
            { name: "Who said what", detail: "A swap test now rejects memories about the wrong person, but most such trap questions are still answered rather than declined.", status: "known gap" },
            { name: "Rule-based extraction", detail: "The fact grammar misses phrasings it was not written for.", status: "known gap" },
            { name: "Our own test sets are small", detail: "Written by the same author as the rules and evaluated repeatedly. Public benchmarks carry the weight.", status: "known gap" },
            { name: "Answer-check threshold doesn’t transfer", detail: "Fitted on LoCoMo, it declined too many answerable questions on LongMemEval. Calibration per deployment is now built in; it still needs measuring on fresh data.", status: "known gap" },
            { name: "Answer check costs time", detail: "The QA reader adds latency on CPU. It now stops at the first memory that answers, which roughly halves its cost, and can be switched off.", status: "known gap" },
            { name: "Tenants share term statistics", detail: "The importance score’s word-rarity feature is counted across all users. Fixing it needs a refit of the importance model.", status: "known gap" },
            { name: "Single API process", detail: "One process owns SQLite and the background jobs. Scaling out needs a shared database.", status: "known gap" },
            { name: "Deleted vectors linger", detail: "They stay in the graph, filtered out, until a rebuild (python -m backend.tools.reindex --force).", status: "known gap" },
        ],
    },
    {
        title: "Next",
        lede: "Planned work, in rough order.",
        items: [
            { name: "Implicit facts", detail: "Infer what a statement implies (a sister’s wedding means she is married) with commonsense rules checked by an entailment model.", status: "next" },
            { name: "A trained extractor", detail: "Distil an LLM’s understanding of phrasing into a small, deterministic model that ships as a file.", status: "next" },
            { name: "Faster imports", detail: "Batched storing and a persistent embedding cache for large histories.", status: "next" },
            { name: "Parallel index build", detail: "Multi-threaded HNSW insertion for faster rebuilds.", status: "next" },
            { name: "Disk-mapped indexes", detail: "Load indexes instantly instead of rebuilding them at start-up.", status: "next" },
            { name: "Conflicts between agents", detail: "Surface contradictory facts written by different agents.", status: "next" },
        ],
    },
    {
        title: "Exploring",
        lede: "Research directions without a promise attached.",
        items: [
            { name: "Vector compression", detail: "8-bit quantisation to cut memory use at very large scale.", status: "exploring" },
            { name: "Adaptive search effort", detail: "Spend more search effort only on queries that need it.", status: "exploring" },
            { name: "Sharding", detail: "Split indexes across machines.", status: "exploring" },
        ],
    },
];

export default function RoadmapPage() {
    return (
        <main>
            <Nav />
            <Section hero className="text-center" backdrop={<MemoryField />}>
                <ScrollFade>
                    <Reveal from="fade"><Eyebrow>Roadmap</Eyebrow></Reveal>
                    <Headline as="h1">
                        <Reveal as="span" delay={0.08}>What’s done.</Reveal>
                        <Reveal as="span" delay={0.2} className="text-sheen">What isn’t.</Reveal>
                    </Headline>
                    <Reveal delay={0.34}>
                        <Lede className="mx-auto mt-6">The honest state of Recallix — including the parts that still fall short.</Lede>
                    </Reveal>
                </ScrollFade>
            </Section>

            <Section>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
                    {GROUPS.map((g, i) => (
                        <Reveal key={g.title} from={i < 2 ? "left" : "right"} delay={(i % 2) * 0.08}>
                            <div className="rounded-[24px] bg-card p-6 h-full">
                                <p className="type-stat"><CountUp value={String(g.items.length)} duration={1} /></p>
                                <p className="mt-2 text-[15px] text-ink-2">{g.title.toLowerCase()}</p>
                            </div>
                        </Reveal>
                    ))}
                </div>
            </Section>

            {GROUPS.map((g, gi) => (
                <Section key={g.title} tone={gi % 2 === 0 ? "cloud" : "paper"}>
                    <div className="grid gap-10 md:grid-cols-[1fr_2fr]">
                        <Reveal from="left">
                            <Headline>{g.title}.</Headline>
                            <Lede className="mt-4">{g.lede}</Lede>
                        </Reveal>
                        <div className={`rounded-[28px] card-edge divide-y divide-line/60 overflow-hidden`}>
                            {g.items.map((it, ii) => (
                                <Reveal key={it.name} from="right" delay={Math.min(ii, 6) * 0.05}><div className="px-6 md:px-8 py-5 flex gap-4 items-start">
                                    <div className="flex-1">
                                        <h3 className="text-[17px] font-semibold">{it.name}</h3>
                                        <p className="text-[15px] text-ink-2 mt-1">{it.detail}</p>
                                    </div>
                                    <span className={`shrink-0 rounded-full px-3 py-1 text-[12px] font-medium ${STATUS_STYLE[it.status]}`}>{it.status}</span>
                                </div></Reveal>
                            ))}
                        </div>
                    </div>
                </Section>
            ))}

            <ClosingCTA />

            <Footer />
        </main>
    );
}
