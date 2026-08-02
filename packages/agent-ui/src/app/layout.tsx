// Root layout is a passthrough; <html>/<body> live in [locale]/layout.tsx so the
// lang attribute follows the active locale (next-intl App Router pattern).
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return children;
}
