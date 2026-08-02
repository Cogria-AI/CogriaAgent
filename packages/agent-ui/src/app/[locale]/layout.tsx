import { NextIntlClientProvider, hasLocale } from 'next-intl';
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';

import '../globals.css';
import AgentShell from '@/components/AgentShell';
import { APP_DESCRIPTION, APP_TITLE } from '@/lib/config';
import { locales } from '@/i18n';

export const metadata: Metadata = { title: APP_TITLE, description: APP_DESCRIPTION };

export function generateStaticParams() {
  return locales.map((locale) => ({ locale }));
}

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  if (!hasLocale(locales, locale)) notFound();

  return (
    <html lang={locale}>
      <body className="antialiased">
        <NextIntlClientProvider>
          <AgentShell>{children}</AgentShell>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
