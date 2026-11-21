import "@fontsource/fragment-mono/400.css";
import "@fontsource/schibsted-grotesk/400.css";
import "@fontsource/schibsted-grotesk/500.css";
import "@fontsource/schibsted-grotesk/600.css";
import "@fontsource/schibsted-grotesk/700.css";
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "faro-cv · Adversarial Vision & Signal Purification Workbench",
  description:
    "Precision interactive workbench for adversarial attacks, Truncated SVD, 2D-FFT signal purification, and FSM-governed computer vision robustness experiments.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans bg-lab-bg text-lab-text selection:bg-lab-cutoff/25 selection:text-white">
        {children}
      </body>
    </html>
  );
}
