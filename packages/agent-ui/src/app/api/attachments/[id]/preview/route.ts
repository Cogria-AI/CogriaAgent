/**
 * Attachment bytes, proxied for previews and downloads.
 *
 * Always goes through agentserv with the caller's own JWT, so a leaked
 * attachment id is useless to anyone else. Responses are forced to download
 * semantics: a user-supplied file rendered inline on this origin would be an
 * XSS vector (SVG and HTML most obviously), and no thumbnail is worth that.
 */
import { NextRequest, NextResponse } from 'next/server';

import { ExchangeError, getOrExchangeJwt } from '@/lib/auth';
import { AGENTSERV_URL } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const INLINE_SAFE = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif']);

export async function GET(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  if (!/^att_[a-z0-9]+$/i.test(id)) {
    return NextResponse.json({ error: 'bad id' }, { status: 400 });
  }

  let jwt;
  try {
    jwt = await getOrExchangeJwt(request);
  } catch (err) {
    if (err instanceof ExchangeError) {
      return NextResponse.json({ error: 'exchange_failed' }, { status: 502 });
    }
    throw err;
  }
  if (!jwt) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });

  const upstream = await fetch(`${AGENTSERV_URL}/attachments/${id}/raw`, {
    headers: { Authorization: `Bearer ${jwt.token}` },
    cache: 'no-store',
  });
  if (!upstream.ok || !upstream.body) {
    return new NextResponse(null, { status: upstream.status });
  }

  const type = upstream.headers.get('content-type') || 'application/octet-stream';
  // Raster images are safe to render in an <img>; everything else downloads.
  const disposition = INLINE_SAFE.has(type) ? 'inline' : 'attachment';
  return new NextResponse(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': type,
      'Content-Disposition': `${disposition}; filename="attachment"`,
      'X-Content-Type-Options': 'nosniff',
      'Content-Security-Policy': "default-src 'none'; img-src 'self' data:; sandbox",
      'Cache-Control': 'private, max-age=300',
    },
  });
}
