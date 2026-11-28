import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Architecture",
    description: "How Recallic is built: a C++ vector engine, a Python memory pipeline and SQLite, with no LLM in the loop.",
    alternates: { canonical: "/architecture" },
    openGraph: { title: "Architecture · Recallic", description: "How Recallic is built: a C++ vector engine, a Python memory pipeline and SQLite, with no LLM in the loop.", url: "/architecture", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Architecture · Recallic", description: "How Recallic is built: a C++ vector engine, a Python memory pipeline and SQLite, with no LLM in the loop.", images: ["/twitter-image.png"] },
};

export default function Layout({ children }: { children: React.ReactNode }) {
    return children;
}
