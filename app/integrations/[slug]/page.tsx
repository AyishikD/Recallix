import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Nav } from "@/components/site/Nav";
import { Footer } from "@/components/site/Footer";
import { ButtonLink, CodeBlock, DataTable, Eyebrow, Headline, Lede, Section } from "@/components/site/ui";
import { INTEGRATIONS, integration } from "@/lib/integrations";

export const dynamicParams = false;  // only the integrations listed in lib/integrations.ts

export function generateStaticParams() {
    return INTEGRATIONS.map((i) => ({ slug: i.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
    const i = integration((await params).slug);
    if (!i) return {};
    return {
        title: i.title,
        description: i.description,
        alternates: { canonical: `/integrations/${i.slug}` },
        openGraph: { title: `${i.title} · Recallic`, description: i.description, url: `/integrations/${i.slug}`, images: ["/opengraph-image.png"] },
        twitter: { card: "summary_large_image", title: `${i.title} · Recallic`, description: i.description, images: ["/twitter-image.png"] },
    };
}

export default async function IntegrationPage({ params }: { params: Promise<{ slug: string }> }) {
    const i = integration((await params).slug);
    if (!i) notFound();
    return (
        <main>
            <Nav />
            <Section hero>
                <Eyebrow>{i.name}</Eyebrow>
                <Headline as="h1">{i.title}</Headline>
                <Lede className="mt-6">{i.lede}</Lede>
                <p className="mt-6 font-mono text-[15px] text-ink">{i.install}</p>
            </Section>
            <Section tone="cloud">
                <CodeBlock label={i.codeLabel} code={i.code} />
                <ul className="mt-10 grid gap-4 md:grid-cols-2">
                    {i.points.map((p) => (
                        <li key={p} className="rounded-[20px] card-edge p-6 text-[16px] leading-relaxed text-ink">{p}</li>
                    ))}
                </ul>
            </Section>
            <Section>
                <Headline>Settings</Headline>
                <div className="mt-8">
                    <DataTable head={["Option", "Default", "What it does"]}
                        rows={i.settings.map(([opt, def, what]) => [<code key="o">{opt}</code>, def, what])} />
                </div>
                <p className="mt-8 text-[15px] text-ink-2 max-w-[720px]">
                    Needs a running Recallic server: <a className="underline" href="/sdk">set one up</a> with Docker, or deploy it to your
                    own Render account in one click. Nothing goes through us; your data stays on your server.
                </p>
                <div className="mt-8 flex flex-wrap gap-6">
                    <ButtonLink href={i.readme} variant="link">Full guide</ButtonLink>
                    {i.registry && <ButtonLink href={i.registry.href} variant="link">{i.registry.label}</ButtonLink>}
                    <ButtonLink href="/integrations" variant="link">All integrations</ButtonLink>
                </div>
            </Section>
            <Footer />
        </main>
    );
}
