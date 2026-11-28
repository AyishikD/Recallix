import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Roadmap",
    description: "The honest state of Recallic, including the parts that still fall short.",
    alternates: { canonical: "/roadmap" },
    openGraph: { title: "Roadmap · Recallic", description: "The honest state of Recallic, including the parts that still fall short.", url: "/roadmap", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Roadmap · Recallic", description: "The honest state of Recallic, including the parts that still fall short.", images: ["/twitter-image.png"] },
};

export default function Layout({ children }: { children: React.ReactNode }) {
    return children;
}
