import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
  // Empty turbopack config silences the "webpack config with no turbopack config" warning.
  // All ML libs (Tesseract.js, TF.js, @huggingface/transformers) run fine under Turbopack
  // with no extra configuration needed.
  turbopack: {},
};

export default nextConfig;
