/**
 * Agent JWT exchange / caching (BFF side).
 *
 * The BFF receives the user's session cookie and:
 *   1. Looks up Redis for a cached JWT keyed by {cookie hash, locale}.
 *   2. If missing / near-expiry, POSTs the cookie to EXCHANGE_ENDPOINT (the
 *      business backend) to mint a fresh JWT, then caches it.
 *   3. Returns the JWT for downstream calls to agentserv.
 *
 * The exchange target is always an explicit EXCHANGE_ENDPOINT — no host or path
 * sniffing — and there is no tenant dimension.
 */
import { createHash } from 'node:crypto';

import { NextRequest } from 'next/server';

import { EXCHANGE_ENDPOINT, JWT_CACHE_TTL_SECONDS } from './config';
import { getRedis } from './redis';

export interface ExchangedJwt {
  token: string;
  expires_at: number;
  /** Surfaced by exchange for the user menu; cached with the JWT JSON. */
  user?: { email?: string; role?: string };
}

const REFRESH_WINDOW_SECONDS = 60;

function hashCookie(cookieHeader: string): string {
  return createHash('sha256').update(cookieHeader).digest('hex').slice(0, 16);
}

function cacheKey(cookieHash: string, locale?: string): string {
  return `agent:jwt:${cookieHash}${locale ? `:${locale}` : ''}`;
}

export class ExchangeError extends Error {
  constructor(public status: number, message: string, public payload?: unknown) {
    super(message);
  }
}

/** Resolve a usable JWT for the current session. Returns null only for
 * caller-correctable conditions (401: session expired). Other failures throw. */
export async function getOrExchangeJwt(request: NextRequest, locale?: string): Promise<ExchangedJwt | null> {
  // Forward whatever cookie exists (possibly empty). We don't short-circuit on a
  // missing cookie: the exchange endpoint is the authority — a real backend 401s
  // without a session (BFF returns null -> unauthenticated), while a dev backend
  // may mint a token unconditionally for local demos.
  const cookie = request.headers.get('cookie') || '';

  const key = cacheKey(hashCookie(cookie), locale);
  const redis = getRedis();

  try {
    const cached = await redis.get(key);
    if (cached) {
      const parsed: ExchangedJwt = JSON.parse(cached);
      const remaining = parsed.expires_at - Math.floor(Date.now() / 1000);
      if (remaining > REFRESH_WINDOW_SECONDS) return parsed;
    }
  } catch {
    // best-effort cache; continue to exchange
  }

  if (!EXCHANGE_ENDPOINT) throw new ExchangeError(500, 'EXCHANGE_ENDPOINT not configured');

  const res = await fetch(EXCHANGE_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Cookie: cookie,
      'X-Requested-With': 'XMLHttpRequest',
    },
    body: JSON.stringify(locale ? { locale } : {}),
    redirect: 'manual',
    cache: 'no-store',
  });

  if (res.status === 401) return null;
  if (!res.ok) {
    throw new ExchangeError(res.status, `exchange failed: ${res.status}`, await safeJson(res));
  }

  const payload = (await res.json()) as ExchangedJwt;
  try {
    await redis.set(key, JSON.stringify(payload), 'EX', JWT_CACHE_TTL_SECONDS);
  } catch {
    // best-effort
  }
  return payload;
}

/** Drop any cached JWT for this session — call after agentserv returns
 * 401/expired so the next request re-exchanges. */
export async function invalidateJwtCache(request: NextRequest, locale?: string): Promise<void> {
  const cookie = request.headers.get('cookie') || '';
  try {
    await getRedis().del(cacheKey(hashCookie(cookie), locale));
  } catch {
    // best-effort
  }
}

async function safeJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return undefined;
  }
}
