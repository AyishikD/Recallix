import type { MetadataRoute } from "next";

export const dynamic = "force-static";  // the site is a static export

export default function robots(): MetadataRoute.Robots {
    return {
        rules: { userAgent: "*", allow: "/", disallow: ["/observability"] },
        sitemap: "https://www.recallic.com/sitemap.xml",
        host: "https://www.recallic.com",
    };
}
