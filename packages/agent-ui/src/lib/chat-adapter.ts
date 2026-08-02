/**
 * Custom assistant-ui ChatModelAdapter for the CogriaAgent SSE protocol.
 *
 * Wire format (sent by agentserv, piped through the BFF):
 * Attachments: the attachment adapter uploads files as the message is sent (see
 * lib/attachment-adapter), so a turn carries only `attachment_ids` — never bytes.
 *
 *   event: ready             → { tools: string[] }
 *   event: conversation      → { conversation_id, new }
 *   event: text              → { delta: string }
 *   event: tool_call         → { id, name, args }
 *   event: tool_result       → { name, content }
 *   event: confirm_required  → { proposal_token, summary, action_name }
 *   event: error             → { message }
 *   event: done              → {}
 *
 * Propose/confirm (MVP-A): a write action streams `confirm_required` instead of
 * a final answer. We surface it to the inline ConfirmCard via onConfirmRequired,
 * await the user, then issue a SECOND /api/chat request carrying the
 * proposal_token — its stream continues into the SAME assistant turn.
 *
 * The artifact tool set and the optional long-running-image action are injected
 * by the caller rather than hardcoded here.
 */
import type { ChatModelAdapter, ThreadAssistantMessagePart } from '@assistant-ui/react';

import { artifactKeyFor, imageGenKeyFor, isArtifactTool, type ArtifactToolDef } from '@/artifacts/registry';
import { isUploaded } from '@/lib/attachment-adapter';
import { useArtifactStore } from '@/lib/artifact-store';

interface ConfirmRequest {
  proposal_token: string;
  summary: string;
  action_name?: string;
}

interface ChatAdapterOptions {
  /** Current UI locale; forwarded so the JWT — and thus the reply language —
   * follows the language the user is viewing. */
  locale: string;
  getConversationId: () => number | null;
  onConversation?: (id: number, isNew: boolean) => void;
  onComplete?: () => void;
  onConfirmRequired?: (req: ConfirmRequest) => Promise<{ confirmed: boolean }>;
  /** Active artifact tool set (builtins + project extras). Drives panel routing. */
  artifactTools: ArtifactToolDef[];
  /** Optional long-running ACTION whose result yields {data:{image_url}}; while
   * it runs (post-confirm) we paint a "generating…" placeholder. */
  longRunningImageAction?: string;
}

interface Accumulator {
  textBuffer: string;
  toolCalls: Map<string, { name: string; args: unknown; result?: string }>;
}

export function buildChatAdapter({
  locale,
  getConversationId,
  onConversation,
  onConfirmRequired,
  onComplete,
  artifactTools,
  longRunningImageAction,
}: ChatAdapterOptions): ChatModelAdapter {
  async function postChat(body: Record<string, unknown>, signal: AbortSignal | undefined): Promise<Response> {
    return fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ locale, ...body }),
      signal,
    });
  }

  function render(acc: Accumulator): ThreadAssistantMessagePart[] {
    const parts: ThreadAssistantMessagePart[] = [];
    for (const [id, entry] of acc.toolCalls) {
      parts.push({
        type: 'tool-call',
        toolCallId: id,
        toolName: entry.name,
        args: entry.args as Record<string, unknown>,
        ...(entry.result !== undefined
          ? { result: safeParse(entry.result), argsText: JSON.stringify(entry.args) }
          : { argsText: JSON.stringify(entry.args) }),
      } as ThreadAssistantMessagePart);
    }
    if (acc.textBuffer) {
      parts.push({ type: 'text', text: acc.textBuffer } as ThreadAssistantMessagePart);
    }
    return parts;
  }

  return {
    async *run({ messages, abortSignal }) {
      const last = messages[messages.length - 1];
      const message =
        last?.content
          .map((p) => ('text' in p ? (p as { text: string }).text : ''))
          .filter(Boolean)
          .join('\n') ?? '';

      // Files were uploaded by the attachment adapter as this message was sent;
      // the turn carries only their ids, and agentserv resolves them, checks
      // ownership and injects the content. Anything whose upload failed keeps a
      // placeholder id — it stays visible in the transcript but isn't sent.
      const attachmentIds = (last?.attachments ?? []).map((a) => a.id).filter(isUploaded);

      const acc: Accumulator = { textBuffer: '', toolCalls: new Map() };

      let resp = await postChat(
        {
          conversation_id: getConversationId(),
          message,
          ...(attachmentIds.length ? { attachment_ids: attachmentIds } : {}),
        },
        abortSignal
      );

      let confirmedTurn = false;
      const imageGenCalls = new Map<string, string>(); // toolCallId -> artifact key

      while (true) {
        if (!resp.ok || !resp.body) {
          const text = await resp.text().catch(() => '');
          acc.textBuffer += `\n\n[error] Request failed (${resp.status}): ${text || 'no body'}`;
          yield { content: render(acc) };
          return;
        }

        let confirm: ConfirmRequest | null = null;

        for await (const frame of parseSSE(resp.body)) {
          if (frame.event === 'conversation') {
            const d = frame.data as { conversation_id?: number; new?: boolean };
            if (d.conversation_id) onConversation?.(d.conversation_id, !!d.new);
          } else if (frame.event === 'text') {
            acc.textBuffer += (frame.data as { delta?: string }).delta ?? '';
          } else if (frame.event === 'tool_call') {
            const d = frame.data as { id: string; name: string; args: unknown };
            acc.toolCalls.set(d.id, { name: d.name, args: d.args });
            const args = (d.args ?? {}) as Record<string, unknown>;
            if (isArtifactTool(d.name, artifactTools)) {
              // Artifact tool: the call IS the render instruction (no-op server-side).
              useArtifactStore.getState().upsert({
                id: artifactKeyFor(d.name, args),
                toolName: d.name,
                args,
                status: 'ready',
              });
            } else if (longRunningImageAction && d.name === longRunningImageAction && confirmedTurn) {
              // Long-running action with no URL yet: paint a placeholder; its
              // result (below) upgrades the SAME tab to the real image.
              const key = imageGenKeyFor(d.id);
              imageGenCalls.set(d.id, key);
              useArtifactStore.getState().upsert({
                id: key,
                toolName: 'preview_image',
                args: { prompt: args.prompt },
                status: 'generating',
              });
            }
          } else if (frame.event === 'tool_result') {
            const d = frame.data as { name: string; content: string };
            for (const [cid, entry] of acc.toolCalls) {
              if (entry.name === d.name && entry.result === undefined) {
                entry.result = d.content;
                const key = imageGenCalls.get(cid);
                if (key) {
                  const url = extractImageUrl(d.content);
                  if (url) {
                    const promptArg = (entry.args ?? {}) as Record<string, unknown>;
                    useArtifactStore.getState().upsert({
                      id: key,
                      toolName: 'preview_image',
                      args: { image_url: url, prompt: promptArg.prompt },
                      status: 'ready',
                    });
                  }
                }
                break;
              }
            }
          } else if (frame.event === 'confirm_required') {
            const d = frame.data as ConfirmRequest;
            if (d.proposal_token) confirm = d;
          } else if (frame.event === 'turn_limit') {
            const d = frame.data as { message?: string };
            if (d.message) acc.textBuffer += `\n\n${d.message}`;
          } else if (frame.event === 'error') {
            const d = frame.data as { message?: string };
            acc.textBuffer += `\n\n[error] ${d.message ?? 'unknown'}`;
          }

          yield { content: render(acc) };
        }

        if (!confirm) {
          onComplete?.();
          return;
        }

        if (!onConfirmRequired) return;
        const { confirmed } = await onConfirmRequired(confirm);
        confirmedTurn = confirmed;

        resp = await postChat(
          {
            conversation_id: getConversationId(),
            message: confirmed ? '[CONFIRMED]' : '[CANCELLED]',
            proposal_token: confirm.proposal_token,
            confirmation: confirmed ? 'confirmed' : 'cancelled',
          },
          abortSignal
        );
      }
    },
  };
}

function safeParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}

function extractImageUrl(content: string): string | null {
  try {
    const env = JSON.parse(content) as { data?: { image_url?: unknown } };
    const url = env?.data?.image_url;
    return typeof url === 'string' && url ? url : null;
  } catch {
    return null;
  }
}

/** Parse a `text/event-stream` body into discrete frames. */
async function* parseSSE(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<{ event: string; data: unknown }, void, void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx = buf.indexOf('\n\n');
      while (idx !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const frame = parseFrame(block);
        if (frame) yield frame;
        idx = buf.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(block: string): { event: string; data: unknown } | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const raw of block.split('\n')) {
    const line = raw.replace(/\r$/, '');
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  const raw = dataLines.join('\n');
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: raw };
  }
}
