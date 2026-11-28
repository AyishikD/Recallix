import type { Metadata, Viewport } from "next";
import "./globals.css";

const DESCRIPTION =
  "Persistent, explainable memory for AI agents. Stores what users tell any agent, recalls it by meaning, keeps facts current, and declines when nothing answers. No LLM in the loop.";

export const metadata: Metadata = {
  metadataBase: new URL("https://www.recallic.com"),
  title: { default: "Recallic — Memory for AI agents", template: "%s · Recallic" },
  description: DESCRIPTION,
  applicationName: "Recallic",
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: "Recallic",
    title: "Recallic — Memory for AI agents",
    description: DESCRIPTION,
    url: "/",
  },
  twitter: { card: "summary_large_image", title: "Recallic — Memory for AI agents", description: DESCRIPTION },
  verification: {
    google: "lJONA5RwyoNhm9tYSB1jo9DVZLqjWgQNbM2E7xva0YY",  // the older recallix.vercel.app property; recallic.com is verified by DNS
  },
};

// Structured data: tells search engines who makes the site and what it is (name, logo, links in results).
const STRUCTURED_DATA = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://www.recallic.com/#org",
      name: "Recallic",
      url: "https://www.recallic.com",
      logo: "https://www.recallic.com/brand/recallic-icon-navy.png",
      email: "admin@recallic.com",
      sameAs: ["https://github.com/Recallic"],
    },
    {
      "@type": "WebSite",
      "@id": "https://www.recallic.com/#website",
      name: "Recallic",
      url: "https://www.recallic.com",
      publisher: { "@id": "https://www.recallic.com/#org" },
    },
    {
      "@type": "SoftwareApplication",
      name: "Recallic",
      url: "https://www.recallic.com",
      description: DESCRIPTION,
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Linux, macOS, Docker",
      image: "https://www.recallic.com/brand/recallic-icon-navy.png",
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      publisher: { "@id": "https://www.recallic.com/#org" },
      sameAs: ["https://github.com/Recallic/Recallic", "https://pypi.org/project/recallic/", "https://www.npmjs.com/package/recallic"],
    },
  ],
};

export const viewport: Viewport = {
  themeColor: "#14213D",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(STRUCTURED_DATA).replace(/</g, "\\u003c") }} />
        {children}
      </body>
    </html>
  );
}
