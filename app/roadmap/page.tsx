"use client";

import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { ButtonLink, Eyebrow, Headline, Lede, Reveal, Section } from "@/components/site/ui";

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
        ],
    },
    {
        title: "Known gaps",
        lede: "What does not work well yet. We would rather tell you than have you find out.",
        items: [
            { name: "Who said what", detail: "Questions that attribute something to the wrong person are still often answered instead of declined.", status: "known gap" },
            { name: "Rule-based extraction", detail: "The fact grammar misses phrasings it was not written for.", status: "known gap" },
            { name: "Our own test sets are small", detail: "Written by the same author as the rules and evaluated repeatedly. Public benchmarks carry the weight.", status: "known gap" },
            { name: "Answer check costs time", detail: "The QA reader adds latency to each recall on CPU. It can be switched off in config.yaml.", status: "known gap" },
            { name: "Single API process", detail: "One process owns SQLite and the background jobs. Scaling out needs a shared database.", status: "known gap" },
            { name: "Deleted vectors linger", detail: "They stay in the graph, filtered out, until a rebuild (python -m backend.tools.reindex --force).", status: "known gap" },
        ],
    },
    {
        title: "Next",
        lede: "Planned work, in rough order.",
        items: [
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
            <Section className="text-center">
                <Reveal>
                    <Eyebrow>Roadmap</Eyebrow>
                    <Headline as="h1">What’s done.<br />What isn’t.</Headline>
                    <Lede className="mx-auto mt-6">The honest state of Recallix — including the parts that still fall short.</Lede>
                </Reveal>
            </Section>

            {GROUPS.map((g, gi) => (
                <Section key={g.title} tone={gi % 2 === 0 ? "cloud" : "paper"}>
                    <div className="grid gap-10 md:grid-cols-[1fr_2fr]">
                        <Reveal>
                            <Headline>{g.title}.</Headline>
                            <Lede className="mt-4">{g.lede}</Lede>
                        </Reveal>
                        <div className={`rounded-[28px] ${gi % 2 === 0 ? "bg-white" : "bg-cloud"} divide-y divide-line/60 overflow-hidden`}>
                            {g.items.map(it => (
                                <div key={it.name} className="px-6 md:px-8 py-5 flex gap-4 items-start">
                                    <div className="flex-1">
                                        <h3 className="text-[17px] font-semibold">{it.name}</h3>
                                        <p className="text-[15px] text-ink-2 mt-1">{it.detail}</p>
                                    </div>
                                    <span className={`shrink-0 rounded-full px-3 py-1 text-[12px] font-medium ${STATUS_STYLE[it.status]}`}>{it.status}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </Section>
            ))}

            <Section tone="night" className="text-center">
                <Reveal>
                    <Headline>Open source. Open about it.</Headline>
                    <Lede dark className="mx-auto mt-5">Issues and pull requests are welcome on every item above.</Lede>
                    <div className="mt-8"><ButtonLink href="https://github.com/AyishikD/Recallix">View on GitHub</ButtonLink></div>
                </Reveal>
            </Section>

            <Footer />
        </main>
    );
}
