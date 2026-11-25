import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", weight: ["400", "500", "600", "700"] });

export const metadata: Metadata = {
  title: "Agent Studio",
  description: "Watch the Cardinal agent work: every search, read, command and check, live.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <body className={mono.variable}>{children}</body>
    </html>
  );
}
