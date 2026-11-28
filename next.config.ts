import type { NextConfig } from "next";

// Static export, deployed by Vercel (https://www.recallic.com) on every push to main.
const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
