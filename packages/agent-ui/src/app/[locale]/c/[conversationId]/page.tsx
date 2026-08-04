/**
 * Deep link to one conversation.
 *
 * ChatThread does a soft `history.replaceState` to /{locale}/c/{id} after the
 * first message, so a hard refresh lands here — this route has to exist or that
 * refresh 404s.
 *
 * History is fetched on the server rather than by the client after mount: the
 * chat runtime takes its messages at construction time, so a fetch that resolves
 * later would leave the thread permanently empty.
 */
import { cookies } from 'next/headers';

import ChatThread from '@/components/ChatThread';
import { exchangeForCookie } from '@/lib/auth';
import { fetchConversationMessages } from '@/lib/conversations';

export const dynamic = 'force-dynamic';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default async function ConversationPage({
  params,
}: {
  params: Promise<{ locale: string; conversationId: string }>;
}) {
  const { locale, conversationId } = await params;

  // Conversation ids are UUIDs. Anything else — a stale bookmark, a hand-typed
  // URL — opens a fresh thread rather than passing arbitrary input downstream.
  if (!UUID.test(conversationId)) {
    return <ChatThread locale={locale} conversationId={null} />;
  }

  const cookieHeader = (await cookies()).toString();
  let jwt = null;
  try {
    jwt = await exchangeForCookie(cookieHeader, locale);
  } catch {
    jwt = null; // the composer is what asks them to sign in
  }

  const history = jwt
    ? await fetchConversationMessages(jwt.token, conversationId)
    : ({ ok: false, reason: 'unauthenticated' } as const);

  return (
    <ChatThread
      locale={locale}
      conversationId={conversationId}
      // Signed out, or an id that isn't theirs: an empty thread rather than an
      // error page. The kernel returns the same 404 either way by design.
      initialMessages={history.ok ? history.data.messages : []}
      activeRun={history.ok && history.data.active}
    />
  );
}
