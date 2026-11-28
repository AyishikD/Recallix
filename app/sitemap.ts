import type { MetadataRoute } from "next";
import { INTEGRATIONS } from "@/lib/integrations";

export const dynamic = "force-static";  // the site is a static export

const PAGES = ["", "/demo", "/sdk", "/integrations", ...INTEGRATIONS.map((i) => `/integrations/${i.slug}`),
    "/benchmark", "/architecture", "/flow-control", "/roadmap"];

export default function sitemap(): MetadataRoute.Sitemap {
    return PAGES.map((p) => ({ url: `https://www.recallic.com${p}`, changeFrequency: "weekly", priority: p === "" ? 1 : 0.7 }));
}
