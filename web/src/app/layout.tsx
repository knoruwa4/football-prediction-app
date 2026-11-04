import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Touchline · Football forecasts",
  description: "Evidence-led football match probabilities and model performance.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en"><body>{children}</body></html>
  );
}
