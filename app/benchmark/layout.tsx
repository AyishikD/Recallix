import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Benchmarks",
    description: "Held-out results on LoCoMo and LongMemEval, and a same-questions comparison with an LLM-based memory layer.",
    alternates: { canonical: "/benchmark" },
    openGraph: { title: "Benchmarks · Recallic", description: "Held-out results on LoCoMo and LongMemEval, and a same-questions comparison with an LLM-based memory layer.", url: "/benchmark", images: ["/opengraph-image.png"] },
    twitter: { card: "summary_large_image", title: "Benchmarks · Recallic", description: "Held-out results on LoCoMo and LongMemEval, and a same-questions comparison with an LLM-based memory layer.", images: ["/twitter-image.png"] },
};

export default function Layout({ children }: { children: React.ReactNode }) {
    return children;
}
