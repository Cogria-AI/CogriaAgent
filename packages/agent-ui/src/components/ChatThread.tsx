'use client';

/**
 * Per-conversation runtime + the styled <Thread>. The shell lives in the layout
 * (AgentShell), so navigating conversations swaps only this content.
 *
 * Runtime: custom SSE adapter on a LocalRuntime (not the Vercel AI SDK). The
 * styled <Thread> is runtime-agnostic. De-businessified: no tenant; artifact tool
 * set + optional long-running-image action come from the registry/config.
 */
import { Loader2Icon } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type CompleteAttachment,
  type ThreadAssistantMessagePart,
  type ThreadMessageLike,
} from '@assistant-ui/react';

import { Thread } from '@/components/assistant-ui/thread';
import ConfirmCard, { type PendingConfirmation } from '@/components/ConfirmCard';
import { useConversationRefresh } from '@/components/AgentShell';
import { buildAttachmentAdapter } from '@/lib/attachment-adapter';
import { buildChatAdapter } from '@/lib/chat-adapter';
import type { PersistedMessage } from '@/lib/api';
import { useArtifactStore } from '@/lib/artifact-store';
import {
  ATTACHMENTS_ACCEPT,
  ATTACHMENTS_ENABLED,
  LONG_RUNNING_IMAGE_ACTION,
  MAX_ATTACHMENT_BYTES_CLIENT,
} from '@/lib/config';
import { artifactKeyFor, isArtifactTool, resolveArtifactTools } from '@/artifacts/registry';

interface Props {
  locale: string;
  conversationId: number | null;
  initialMessages?: PersistedMessage[];
}

type ReplayArtifact = { id: string; toolName: string; args: Record<string, unknown>; status: 'ready' };

const ARTIFACT_TOOLS = resolveArtifactTools();

function artifactsForRow(m: PersistedMessage): ReplayArtifact[] {
  if (m.role !== 'assistant' || !m.content.tool_calls) return [];
  const out: ReplayArtifact[] = [];
  for (const tc of m.content.tool_calls) {
    const args = (tc.args ?? {}) as Record<string, unknown>;
    if (isArtifactTool(tc.name, ARTIFACT_TOOLS)) {
      out.push({ id: artifactKeyFor(tc.name, args), toolName: tc.name, args, status: 'ready' });
    }
  }
  return out;
}

export function artifactsFromHistory(rows: PersistedMessage[]): ReplayArtifact[] {
  const byKey = new Map<string, ReplayArtifact>();
  for (const m of rows) for (const a of artifactsForRow(m)) byKey.set(a.id, a);
  return [...byKey.values()];
}

/** Persisted attachment refs -> the shape assistant-ui renders as chips. The
 * bytes are fetched lazily by the preview route, keyed on the same id. */
function attachmentsForRow(m: PersistedMessage): CompleteAttachment[] {
  return (m.content.attachments ?? []).map((a) => ({
    id: a.id,
    type: a.kind === 'image' ? 'image' : 'document',
    name: a.name,
    contentType: a.mime,
    status: { type: 'complete' },
    content: [],
  }));
}

function toThreadMessages(rows: PersistedMessage[]): ThreadMessageLike[] {
  const out: ThreadMessageLike[] = [];
  for (const m of rows) {
    if (m.role === 'user') {
      out.push({
        role: 'user',
        content: [{ type: 'text', text: m.content.text || '' }],
        attachments: attachmentsForRow(m),
      });
    } else if (m.role === 'assistant') {
      const parts: ThreadAssistantMessagePart[] = [];
      for (const a of artifactsForRow(m)) {
        parts.push({
          type: 'tool-call',
          toolCallId: a.id,
          toolName: a.toolName,
          args: a.args,
          argsText: JSON.stringify(a.args),
          result: { replayed: true },
        } as ThreadAssistantMessagePart);
      }
      const text = m.content.text || '';
      if (text) parts.push({ type: 'text', text } as ThreadAssistantMessagePart);
      if (parts.length) out.push({ role: 'assistant', content: parts });
    }
  }
  return out;
}

/** Files upload when the message is sent, and assistant-ui clears the composer
 * before the upload resolves — without this the UI would sit silent for however
 * long a large PDF takes to store and extract. */
function UploadingHint({ count }: { count: number }) {
  const t = useTranslations('chat');
  return (
    <div className="mx-auto w-full max-w-(--thread-max-width) px-2">
      <div className="flex items-center gap-2 text-muted-foreground text-sm">
        <Loader2Icon className="size-3.5 animate-spin" aria-hidden />
        {t('uploading', { count })}
      </div>
    </div>
  );
}

/** Upload refusals (too big, wrong type, unreadable scan) come back as plain
 * sentences from agentserv — show them as-is rather than a generic failure. */
function AttachmentError({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="mx-auto flex w-full max-w-(--thread-max-width) items-start gap-2 px-2 text-sm"
    >
      <div className="flex w-full items-start justify-between gap-3 rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-destructive">
        <span>{message}</span>
        <button type="button" onClick={onDismiss} className="shrink-0 underline" aria-label="Dismiss">
          ✕
        </button>
      </div>
    </div>
  );
}

export default function ChatThread({ locale, conversationId, initialMessages }: Props) {
  const { refreshConversations } = useConversationRefresh();
  const convIdRef = useRef<number | null>(conversationId);

  const hydrateArtifacts = useArtifactStore((s) => s.hydrate);
  useEffect(() => {
    const key = conversationId != null ? `c:${conversationId}` : 'new';
    const seed = conversationId != null && initialMessages ? artifactsFromHistory(initialMessages) : [];
    hydrateArtifacts(key, seed);
  }, [conversationId, initialMessages, hydrateArtifacts]);

  const [pending, setPending] = useState<PendingConfirmation | null>(null);

  const onConversation = useCallback(
    (id: number, isNew: boolean) => {
      convIdRef.current = id;
      if (isNew) {
        // Soft URL swap; avoid router navigation which would unmount this and
        // tear down the live SSE stream mid-reply.
        window.history.replaceState(null, '', `/${locale}/c/${id}`);
      }
      refreshConversations();
    },
    [locale, refreshConversations]
  );

  const onConfirmRequired = useCallback(
    (req: { proposal_token: string; summary: string; action_name?: string }) =>
      new Promise<{ confirmed: boolean }>((resolve) => {
        setPending({
          token: req.proposal_token,
          summary: req.summary,
          actionName: req.action_name,
          resolve: (confirmed: boolean) => {
            setPending(null);
            resolve({ confirmed });
          },
        });
      }),
    []
  );

  const adapter = useMemo(
    () =>
      buildChatAdapter({
        locale,
        getConversationId: () => convIdRef.current,
        onConversation,
        onConfirmRequired,
        onComplete: refreshConversations,
        artifactTools: ARTIFACT_TOOLS,
        longRunningImageAction: LONG_RUNNING_IMAGE_ACTION || undefined,
      }),
    [locale, onConversation, onConfirmRequired, refreshConversations]
  );

  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [uploadingCount, setUploadingCount] = useState(0);
  const attachments = useMemo(
    () =>
      ATTACHMENTS_ENABLED
        ? buildAttachmentAdapter({
            locale,
            accept: ATTACHMENTS_ACCEPT,
            maxBytes: MAX_ATTACHMENT_BYTES_CLIENT,
            onError: setAttachmentError,
            onUploadingChange: setUploadingCount,
          })
        : undefined,
    [locale]
  );

  const runtime = useLocalRuntime(adapter, {
    initialMessages: initialMessages ? toThreadMessages(initialMessages) : undefined,
    ...(attachments ? { adapters: { attachments } } : {}),
  });

  // Bridge artifact-panel action buttons into the conversation as a user message.
  useEffect(() => {
    const onSend = (e: Event) => {
      const text = (e as CustomEvent).detail?.text;
      if (typeof text === 'string' && text.trim()) runtime.thread.append(text);
    };
    window.addEventListener('agent:send', onSend);
    return () => window.removeEventListener('agent:send', onSend);
  }, [runtime]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread
        belowMessages={
          <>
            {uploadingCount > 0 && <UploadingHint count={uploadingCount} />}
            {attachmentError && (
              <AttachmentError message={attachmentError} onDismiss={() => setAttachmentError(null)} />
            )}
            {pending ? <ConfirmCard key={pending.token} pending={pending} /> : null}
          </>
        }
      />
    </AssistantRuntimeProvider>
  );
}
