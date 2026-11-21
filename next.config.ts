import type { NextConfig } from "next";

// Static export for GitHub Pages. The site is served from a sub-path
// (https://<user>.github.io/<repo>/), which the deploy workflow passes in as
// PAGES_BASE_PATH; locally it is empty so `npm run dev` serves from /.
const basePath = process.env.PAGES_BASE_PATH || "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
