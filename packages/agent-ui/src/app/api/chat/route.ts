/**
 * Chat BFF — bridges the browser to agentserv with a JWT in the middle.
 *
 *   browser -> POST /api/chat (cookie + { message, ... })
 *           -> getOrExchangeJwt (Redis cached) -> business backend exchange
 *           -> POST agentserv /chat with Authorization: Bearer
 *           -> SSE stream piped back to the browser unchanged
 *
 * Artifact tools come from the front-end registry and are forwarded here, so
 * agentserv can advertise them to the LLM without knowing what they render.
 */
import { NextRequest, NextResponse } from 'next/server';

import { resolveArtifactTools } from '@/artifacts/registry';
import { ExchangeError, getOrExchangeJwt, invalidateJwtCache } from '@/lib/auth';
import { AGENTSERV_URL } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface ChatBody {
  conversation_id?: number | null;
  message?: string;
  locale?: string;
  proposal_token?: string;
  confirmation?: 'confirmed' | 'cancelled';
  attachment_ids?: string[];
}

export async function POST(request: NextRequest) {
  let body: ChatBody;
  try {
    body = (await request.json()) as ChatBody;
  } catch {
    return NextResponse.json({ error: 'invalid json' }, { status: 400 });
  }

  const message = (body.message || '').trim();
  if (!message) {
    return NextResponse.json({ error: 'message required' }, { status: 422 });
  }
  const conversationId = body.conversation_id ?? null;
  const locale = (body.locale || '').trim() || undefined;

  const upstreamBody: Record<string, unknown> = { conversation_id: conversationId, message };
  if (body.proposal_token) upstreamBody.proposal_token = body.proposal_token.trim();
  // Ids only — the bytes were uploaded via /api/attachments and belong to the
  // caller; agentserv re-checks ownership before injecting anything.
  if (Array.isArray(body.attachment_ids) && body.attachment_ids.length) {
    upstreamBody.attachment_ids = body.attachment_ids.filter((id) => typeof id === 'string' && id);
  }
  if (body.confirmation) upstreamBody.confirmation = body.confirmation;
  // Artifact tools live in the front-end registry; tell agentserv so it can
  // advertise them to the LLM and dispatch them as no-ops.
  upstreamBody.artifact_tools = resolveArtifactTools();

  // One built-in retry: if agentserv returns 401 token_expired despite our cache
  // saying it's valid (clock drift / stale), invalidate + re-exchange.
  for (let attempt = 0; attempt < 2; attempt++) {
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
    if (!jwt) {
      return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });
    }

    const upstream = await fetch(AGENTSERV_URL + '/chat', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${jwt.token}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(upstreamBody),
      cache: 'no-store',
    });

    if (upstream.status === 401 && attempt === 0) {
      const text = await upstream.text().catch(() => '');
      if (/token expired/i.test(text)) {
        await invalidateJwtCache(request, locale);
        continue;
      }
      return new NextResponse(text || 'unauthorized', { status: 401 });
    }

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

  return NextResponse.json({ error: 'exhausted retries' }, { status: 500 });
}
