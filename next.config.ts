import type { NextConfig } from "next";

// Static export, deployed by Vercel (https://recallix.vercel.app) on every push to main.
const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
