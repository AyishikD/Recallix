import type { Metadata } from "next";

const DESCRIPTION = "Try Recallic in your browser: tell it a few messages, ask questions, and watch it answer from what you said, or decline when nothing answers. No LLM involved.";

export const metadata: Metadata = {
    title: "Try it",
    description: DESCRIPTION,
    alternates: { canonical: "/demo" },
    openGraph: { title: "Try Recallic", description: DESCRIPTION, url: "/demo", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Try Recallic", description: DESCRIPTION, images: ["/twitter-image.png"] },
};

export default function Layout({ children }: { children: React.ReactNode }) {
    return children;
}
