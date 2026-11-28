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

export const viewport: Viewport = {
  themeColor: "#14213D",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
