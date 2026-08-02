/**
 * Attachment upload BFF — the browser's only way to put a file in front of the
 * agent.
 *
 *   browser -> POST /api/attachments (cookie + multipart)
 *           -> getOrExchangeJwt -> business backend exchange
 *           -> POST agentserv /attachments with Authorization: Bearer
 *           <- { id, filename, mime, size_bytes, kind, status, … }
 *
 * The id that comes back is what the composer later sends in `attachment_ids`.
 * Ownership lives in the JWT `sub`, so agentserv can refuse a file that isn't
 * the caller's; this route adds a size gate so an oversized body is rejected
 * here rather than travelling on.
 */
import { NextRequest, NextResponse } from 'next/server';

import { ExchangeError, getOrExchangeJwt } from '@/lib/auth';
import { AGENTSERV_URL, MAX_ATTACHMENT_BYTES } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: 'expected multipart/form-data' }, { status: 400 });
  }

  const file = form.get('file');
  if (!(file instanceof File)) {
    return NextResponse.json({ error: 'file is required' }, { status: 422 });
  }
  if (file.size > MAX_ATTACHMENT_BYTES) {
    const mb = Math.round(MAX_ATTACHMENT_BYTES / (1024 * 1024));
    return NextResponse.json({ error: `That file is larger than the ${mb} MB limit.` }, { status: 413 });
  }

  const locale = (form.get('locale') as string | null)?.trim() || undefined;

  let jwt;
  try {
    jwt = await getOrExchangeJwt(request, locale);
  } catch (err) {
    if (err instanceof ExchangeError) {
      return NextResponse.json({ error: 'exchange_failed', status: err.status }, { status: 502 });
    }
    throw err;
  }
  if (!jwt) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });

  const upstreamForm = new FormData();
  upstreamForm.append('file', file, file.name);

  const upstream = await fetch(`${AGENTSERV_URL}/attachments`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${jwt.token}`, Accept: 'application/json' },
    body: upstreamForm,
    cache: 'no-store',
  });

  const body = await upstream.text();
  if (!upstream.ok) {
    // agentserv's rejections are written for humans ("larger than the 20 MB
    // limit", "this file type isn't supported") — pass them through verbatim.
    return NextResponse.json({ error: extractDetail(body) }, { status: upstream.status });
  }
  return new NextResponse(body, {
    status: 200,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  });
}

function extractDetail(body: string): string {
  try {
    const parsed = JSON.parse(body) as { detail?: unknown; error?: unknown };
    const detail = parsed.detail ?? parsed.error;
    if (typeof detail === 'string' && detail) return detail;
  } catch {
    // fall through
  }
  return body.slice(0, 300) || 'upload failed';
}
