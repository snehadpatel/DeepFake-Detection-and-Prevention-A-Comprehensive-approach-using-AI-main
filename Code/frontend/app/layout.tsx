import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DeepShield AI | Deepfake Detection & Prevention",
  description: "State-of-the-art AI for detecting and preventing deepfake manipulation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-slate-950 text-slate-100 min-h-screen selection:bg-indigo-500/30`}>
        <div className="fixed inset-0 -z-10 h-full w-full bg-slate-950 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:32px_32px] opacity-20"></div>
        <div className="fixed top-0 left-0 -z-10 h-full w-full bg-[radial-gradient(circle_farthest-side_at_0%_0%,#1e1b4b,transparent)] opacity-40"></div>
        <div className="fixed bottom-0 right-0 -z-10 h-full w-full bg-[radial-gradient(circle_farthest-side_at_100%_100%,#1e1b4b,transparent)] opacity-40"></div>
        {children}
      </body>
    </html>
  );
}
