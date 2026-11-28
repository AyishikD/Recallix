import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "How it works",
    description: "Follow messages through Recallic's store and recall pipeline, stage by stage, with measured timings.",
    alternates: { canonical: "/flow-control" },
    openGraph: { title: "How it works · Recallic", description: "Follow messages through Recallic's store and recall pipeline, stage by stage, with measured timings.", url: "/flow-control", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "How it works · Recallic", description: "Follow messages through Recallic's store and recall pipeline, stage by stage, with measured timings.", images: ["/twitter-image.png"] },
};

export default function Layout({ children }: { children: React.ReactNode }) {
    return children;
}
