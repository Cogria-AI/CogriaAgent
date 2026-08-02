import { getRequestConfig } from 'next-intl/server';

import { DEFAULT_LOCALE, SUPPORTED_LOCALES } from './lib/config';

export const locales = SUPPORTED_LOCALES;
export const defaultLocale = DEFAULT_LOCALE;

export function isSupportedLocale(locale: string): boolean {
  return SUPPORTED_LOCALES.includes(locale);
}

export default getRequestConfig(async ({ requestLocale }) => {
  const requested = (await requestLocale) || defaultLocale;
  // Always load a real bundle; unsupported locales fall back to the default so
  // next-intl's tree stays usable inside any placeholder page.
  const messagesLocale = isSupportedLocale(requested) ? requested : defaultLocale;
  return {
    locale: requested,
    messages: (await import(`./messages/${messagesLocale}.json`)).default,
  };
});
