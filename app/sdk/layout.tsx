import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Developers",
    description: "Python and TypeScript SDKs over a small REST API, with integrations for Open WebUI, Google ADK, Haystack, LlamaIndex and the Vercel AI SDK.",
    alternates: { canonical: "/sdk" },
    openGraph: { title: "Developers · Recallic", description: "Python and TypeScript SDKs over a small REST API, with integrations for Open WebUI, Google ADK, Haystack, LlamaIndex and the Vercel AI SDK.", url: "/sdk", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Developers · Recallic", description: "Python and TypeScript SDKs over a small REST API, with integrations for Open WebUI, Google ADK, Haystack, LlamaIndex and the Vercel AI SDK.", images: ["/twitter-image.png"] },
};

export default function Layout({ children }: { children: React.ReactNode }) {
    return children;
}
