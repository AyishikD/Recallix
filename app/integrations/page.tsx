import type { Metadata } from "next";
import Link from "next/link";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { Eyebrow, Headline, Lede, Section } from "@/components/site/ui";
import { INTEGRATIONS } from "@/lib/integrations";

const DESCRIPTION = "Recallic plugs into Open WebUI, LlamaIndex, the Vercel AI SDK, Haystack, Google ADK and any MCP client (Claude Desktop, Cursor) as self-hosted long-term memory that says when it doesn't know.";

export const metadata: Metadata = {
    title: "Integrations",
    description: DESCRIPTION,
    alternates: { canonical: "/integrations" },
    openGraph: { title: "Integrations · Recallic", description: DESCRIPTION, url: "/integrations", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Integrations · Recallic", description: DESCRIPTION, images: ["/twitter-image.png"] },
};

export default function IntegrationsPage() {
    return (
        <main>
            <Nav />
            <Section hero>
                <Eyebrow>Integrations</Eyebrow>
                <Headline as="h1">Memory for the tools you already use.</Headline>
                <Lede className="mt-6">Each one is a small, tested package over the same self-hosted Recallic server. No LLM is called to store or recall memory, and when nothing answers, your model is told so.</Lede>
            </Section>
            <Section tone="cloud">
                <ul className="grid gap-5 md:grid-cols-2">
                    {INTEGRATIONS.map((i) => (
                        <li key={i.slug}>
                            <Link href={`/integrations/${i.slug}`} className="block h-full rounded-[24px] card-edge p-7 hover:border-accent transition-colors">
                                <p className="type-eyebrow text-accent mb-2">{i.name}</p>
                                <h2 className="text-[22px] font-semibold text-ink leading-snug">{i.title}</h2>
                                <p className="mt-3 text-[15px] text-ink-2 leading-relaxed">{i.lede}</p>
                                <p className="mt-4 font-mono text-[13px] text-ink-2">{i.install}</p>
                            </Link>
                        </li>
                    ))}
                </ul>
            </Section>
            <Footer />
        </main>
    );
}
