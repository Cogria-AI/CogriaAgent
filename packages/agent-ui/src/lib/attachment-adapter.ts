/**
 * AttachmentAdapter for the CogriaAgent upload flow.
 *
 * Picking a file costs nothing: it sits in the composer as a local File. The
 * upload (and, server-side, the text extraction) happens in `send`, so a file
 * that gets attached and then removed never touches the server — no orphan
 * blobs, no extraction spent on a message that was never sent.
 *
 * The cost of that choice is that a refusal arrives after the user has already
 * hit send, so two things matter here:
 *   1. `add` pre-checks size and type locally, which is what most refusals are;
 *   2. `send` NEVER rejects. assistant-ui clears the composer before it awaits
 *      us (BaseComposerRuntimeCore.send), so throwing would take the user's
 *      typed message down with it. A failed upload resolves to a placeholder
 *      that the chat adapter drops and the chip renders as "not uploaded".
 *
 * Files never go to the model from here: the browser holds an id, agentserv
 * holds the bytes and the extracted text.
 */
import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from '@assistant-ui/react';

/** What POST /api/attachments returns (agentserv's public attachment view). */
export interface UploadedAttachment {
  id: string;
  filename: string;
  mime: string;
  size_bytes: number;
  kind: 'image' | 'document';
  status: 'extracting' | 'ready' | 'failed';
  page_count: number | null;
  extracted_chars: number | null;
  error: string | null;
}

export const DEFAULT_ACCEPT = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
  'application/pdf',
  '.docx',
  '.xlsx',
  '.pptx',
  '.txt',
  '.md',
  '.csv',
  '.json',
].join(',');

/** Marks an attachment whose upload failed: kept in the transcript so the user
 * can see what didn't make it, but never sent to agentserv. */
const FAILED_PREFIX = 'unsent_';

export const isUploaded = (id: string): boolean => id.startsWith('att_');
export const isFailedUpload = (id: string): boolean => id.startsWith(FAILED_PREFIX);

export function previewUrl(attachmentId: string): string {
  return `/api/attachments/${attachmentId}/preview`;
}

function attachmentType(contentType: string): 'image' | 'document' | 'file' {
  if (contentType.startsWith('image/')) return 'image';
  if (contentType.startsWith('text/') || contentType === 'application/pdf') return 'document';
  return 'file';
}

/** Mirror of the server's allowlist check, close enough to catch the obvious
 * cases at pick time. The server's magic-byte sniff remains the authority. */
function matchesAccept(file: File, accept: string): boolean {
  const patterns = accept
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  if (!patterns.length) return true;
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  return patterns.some((p) =>
    p.startsWith('.') ? name.endsWith(p) : p.endsWith('/*') ? type.startsWith(p.slice(0, -1)) : type === p
  );
}

/** The refusals worth catching before the user hits send, in their words.
 * Returns null when the file looks fine; the server still has the final say. */
function localRejection(file: File, { accept, maxBytes }: { accept: string; maxBytes: number }): string | null {
  if (file.size > maxBytes) {
    return `${file.name} is larger than the ${Math.round(maxBytes / (1024 * 1024))} MB limit.`;
  }
  if (!matchesAccept(file, accept)) {
    return `${file.name}: this file type isn't supported.`;
  }
  return null;
}

export interface AttachmentAdapterOptions {
  locale: string;
  accept?: string;
  maxBytes?: number;
  /** Called with a human-readable reason when a file is refused. */
  onError?: (message: string) => void;
  /** Number of files currently uploading — drives the "uploading…" hint that
   * covers the gap between hitting send and the message appearing. */
  onUploadingChange?: (count: number) => void;
}

export function buildAttachmentAdapter({
  locale,
  accept = DEFAULT_ACCEPT,
  maxBytes = 20 * 1024 * 1024,
  onError,
  onUploadingChange,
}: AttachmentAdapterOptions): AttachmentAdapter {
  let uploading = 0;

  function track(delta: number) {
    uploading = Math.max(0, uploading + delta);
    onUploadingChange?.(uploading);
  }

  async function upload(file: File): Promise<UploadedAttachment> {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('locale', locale);

    const res = await fetch('/api/attachments', { method: 'POST', body: form });
    if (!res.ok) {
      const detail = await res
        .json()
        .then((b: { error?: string }) => b.error)
        .catch(() => undefined);
      throw new Error(detail || `Upload failed (${res.status})`);
    }
    return (await res.json()) as UploadedAttachment;
  }

  return {
    accept,

    async add({ file }) {
      // Local only — nothing is uploaded until the message is sent.
      const base = {
        id: crypto.randomUUID(),
        type: attachmentType(file.type),
        name: file.name,
        contentType: file.type,
        file,
      };
      const rejection = localRejection(file, { accept, maxBytes });
      if (rejection) {
        onError?.(rejection);
        // Resolve with an error state rather than throwing: the composer's
        // add-attachment button calls us fire-and-forget, so a rejected promise
        // would surface as an unhandled rejection. This shows a struck-through
        // chip the user can remove, and send() skips it.
        return { ...base, status: { type: 'incomplete', reason: 'error' } } satisfies PendingAttachment;
      }
      return {
        ...base,
        status: { type: 'requires-action', reason: 'composer-send' },
      } satisfies PendingAttachment;
    },

    async send(attachment) {
      if (attachment.status?.type === 'incomplete') {
        // Already refused locally — don't spend a round trip proving it again.
        return { ...attachment, id: `${FAILED_PREFIX}${attachment.id}`, status: { type: 'complete' }, content: [] };
      }
      track(1);
      try {
        const record = await upload(attachment.file);
        // A stored-but-unreadable file (a scan, say) still goes to the model —
        // it's told why it can't read it — but tell the user as well.
        if (record.status === 'failed' && record.error) onError?.(record.error);
        return {
          ...attachment,
          // Swap in the server id: from here on it identifies the file
          // everywhere (chat body, preview URL, database row).
          id: record.id,
          name: record.filename,
          contentType: record.mime,
          status: { type: 'complete' },
          // Rendering is the chip's job, not the message body's — what the user
          // typed stays exactly what they typed.
          content: [],
        } satisfies CompleteAttachment;
      } catch (e) {
        // Resolving (not throwing) is deliberate: see the note at the top.
        onError?.(e instanceof Error ? e.message : `${attachment.name} could not be uploaded.`);
        return {
          ...attachment,
          id: `${FAILED_PREFIX}${attachment.id}`,
          status: { type: 'complete' },
          content: [],
        } satisfies CompleteAttachment;
      } finally {
        track(-1);
      }
    },

    async remove(_attachment: Attachment) {
      // Nothing was uploaded yet, so there is nothing to clean up.
    },
  };
}
