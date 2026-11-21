import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        lab: {
          bg: "#090A0C",
          surface: "#101216",
          elevated: "#15181E",
          border: "#22262F",
          borderStrong: "#343B48",
          text: "#E6E8EC",
          muted: "#8B93A3",
          subtle: "#596070",
          signal: "#10B981",
          alert: "#EF4444",
          warn: "#F59E0B",
          cutoff: "#38BDF8",
        },
      },
      fontFamily: {
        sans: [
          '"Schibsted Grotesk"',
          "-apple-system",
          "BlinkMacSystemFont",
          "Helvetica Neue",
          "system-ui",
          "sans-serif",
        ],
        mono: [
          '"Fragment Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
