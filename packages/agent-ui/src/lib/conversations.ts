/**
 * Conversation history, read server-side.
 *
 * These run on the server (a page render or a BFF route), so they exchange the
 * session cookie for a JWT themselves rather than going through /api. The
 * kernel owns persistence; nothing here caches.
 */
import { AGENTSERV_URL } from '@/lib/config';
import type { PersistedMessage } from '@/lib/api';

export interface ConversationSummary {
  id: string;
  /** The user-set title if the conversation was renamed, else one the kernel
   * derives from the opening message. */
  title: string;
  message_count: number;
  created_at: string | null;
}

export interface ConversationHistory {
  messages: PersistedMessage[];
  /** A reply is being generated for this conversation right now — the client
   * should re-attach to the resume stream rather than wait for a refresh. */
  active: boolean;
}

/** Distinguishes "signed out" (render the sign-in prompt) from "nothing yet"
 * and from "the id isn't theirs" — each wants a different screen. */
export type HistoryResult<T> =
  | { ok: true; data: T }
  | { ok: false; reason: 'unauthenticated' | 'notFound' | 'error' };

async function agentservGet(jwt: string, path: string): Promise<Response> {
  return fetch(`${AGENTSERV_URL}${path}`, {
    headers: { Authorization: `Bearer ${jwt}`, Accept: 'application/json' },
    cache: 'no-store',
  });
}

function classify(status: number): 'unauthenticated' | 'notFound' | 'error' {
  if (status === 401) return 'unauthenticated';
  if (status === 404) return 'notFound';
  return 'error';
}

export async function fetchConversations(
  jwt: string,
  limit = 50
): Promise<HistoryResult<ConversationSummary[]>> {
  let res: Response;
  try {
    res = await agentservGet(jwt, `/conversations?limit=${limit}`);
  } catch {
    return { ok: false, reason: 'error' };
  }
  if (!res.ok) return { ok: false, reason: classify(res.status) };
  const body = (await res.json()) as { conversations?: ConversationSummary[] };
  return { ok: true, data: body.conversations ?? [] };
}

/**
 * One conversation's messages, shaped for replay. The kernel stores the whole
 * message dict per row and has no per-row id, so the index stands in — the
 * thread only needs stable keys within a single render.
 */
export async function fetchConversationMessages(
  jwt: string,
  conversationId: string
): Promise<HistoryResult<ConversationHistory>> {
  let res: Response;
  try {
    res = await agentservGet(jwt, `/conversations/${encodeURIComponent(conversationId)}`);
  } catch {
    return { ok: false, reason: 'error' };
  }
  if (!res.ok) return { ok: false, reason: classify(res.status) };
  const body = (await res.json()) as {
    messages?: Array<Record<string, unknown>>;
    active?: boolean;
  };
  return {
    ok: true,
    data: {
      messages: (body.messages ?? []).map((m, i) => ({
        id: i,
        role: m.role as PersistedMessage['role'],
        content: (m.content ?? {}) as PersistedMessage['content'],
        created_at: null,
      })),
      active: Boolean(body.active),
    },
  };
}
