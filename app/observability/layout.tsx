    import type { Metadata } from "next";

    export const metadata: Metadata = {
        title: "Dashboard",
        description: "Inspect a user's memories, timeline and reflections in a running Recallic server.",
        alternates: { canonical: "/observability" },
        openGraph: { title: "Dashboard · Recallic", description: "Inspect a user's memories, timeline and reflections in a running Recallic server.", url: "/observability", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Dashboard · Recallic", description: "Inspect a user's memories, timeline and reflections in a running Recallic server.", images: ["/twitter-image.png"] },
robots: { index: false, follow: true },  // a tool for a running server, not a page to find in search
    };

    export default function Layout({ children }: { children: React.ReactNode }) {
        return children;
    }
