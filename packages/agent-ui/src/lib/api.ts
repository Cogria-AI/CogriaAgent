/**
 * Conversation types shared by the front-end. (No tenant param — single-tenant.)
 * For C0 the sidebar/history endpoints aren't wired; agentserv owns persistence.
 * These types describe the persisted message shape used to seed replay.
 */

/** Lightweight reference stored on a user message; the bytes and the extracted
 * text stay in the attachment record server-side. */
export interface PersistedAttachmentRef {
  id: string;
  name: string;
  mime: string;
  size: number;
  kind: 'image' | 'document';
}

export interface PersistedMessage {
  id: number;
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: {
    text?: string;
    attachments?: PersistedAttachmentRef[];
    tool_calls?: Array<{ id: string; name: string; args: unknown }>;
    tool_call_id?: string;
    name?: string;
    result?: unknown;
  };
  created_at: string | null;
}
