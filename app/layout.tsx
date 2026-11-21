import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Recallix — Memory for AI agents",
  description:
    "Persistent, explainable memory for AI agents. Stores what users tell any agent, recalls it by meaning, keeps facts current, and declines when nothing answers. No LLM in the loop.",
  verification: {
    google: "lJONA5RwyoNhm9tYSB1jo9DVZLqjWgQNbM2E7xva0YY",
  },
};

export const viewport: Viewport = {
  themeColor: "#fbfbfd",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
