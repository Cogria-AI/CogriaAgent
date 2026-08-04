/**
 * Resume BFF — re-attach the browser to a reply already being generated.
 *
 *   browser -> GET /api/chat/resume?conversation_id=…&locale=…
 *           -> getOrExchangeJwt -> GET agentserv /conversations/{id}/stream
 *           -> SSE (backlog replay + live tail) piped back unchanged
 *
 * A 404 from upstream means nothing is running — the turn finished while the
 * user was away — and the client refetches history instead of streaming.
 */
import { NextRequest, NextResponse } from 'next/server';

import { ExchangeError, getOrExchangeJwt } from '@/lib/auth';
import { AGENTSERV_URL } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const conversationId = (params.get('conversation_id') || '').trim();
  if (!conversationId) {
    return NextResponse.json({ error: 'conversation_id required' }, { status: 422 });
  }
  const locale = (params.get('locale') || '').trim() || undefined;

  let jwt;
  try {
    jwt = await getOrExchangeJwt(request, locale);
  } catch (err) {
    if (err instanceof ExchangeError) {
      return NextResponse.json(
        { error: 'exchange_failed', status: err.status, detail: err.payload },
        { status: 502 }
      );
    }
    throw err;
  }
  if (!jwt) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });

  const upstream = await fetch(
    `${AGENTSERV_URL}/conversations/${encodeURIComponent(conversationId)}/stream`,
    {
      headers: { Authorization: `Bearer ${jwt.token}`, Accept: 'text/event-stream' },
      cache: 'no-store',
    }
  );

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => '');
    return new NextResponse(text || 'upstream error', { status: upstream.status });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
