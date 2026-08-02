'use client';

/**
 * The little card shown for an attached file — in the composer while it uploads
 * and in the user's message afterwards.
 *
 * Composer previews come from the local File (no round trip); message previews
 * come from /api/attachments/{id}/preview, which proxies the bytes with the
 * caller's own credentials. Only raster images are ever rendered; anything else
 * shows a file glyph.
 */
import { FileTextIcon, ImageIcon, Loader2Icon, XIcon } from 'lucide-react';
import { useEffect, useState, type FC } from 'react';
import { AttachmentPrimitive, useAuiState } from '@assistant-ui/react';

import { isFailedUpload, isUploaded, previewUrl } from '@/lib/attachment-adapter';
import { cn } from '@/lib/utils';

function useLocalPreview(file: File | undefined, isImage: boolean): string | undefined {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!file || !isImage) return;
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file, isImage]);
  return url;
}

export const AttachmentChip: FC<{ removable?: boolean }> = ({ removable = false }) => {
  const attachment = useAuiState((s) => s.attachment);
  const isImage = attachment.type === 'image';
  const uploading = attachment.status?.type === 'running';
  // Either the composer refused it, or the upload failed as the message was
  // sent — in which case the file is in the transcript but not on the server.
  const failed = attachment.status?.type === 'incomplete' || isFailedUpload(attachment.id);
  const local = useLocalPreview(attachment.file, isImage);
  // Composer: preview the local file. Message: the server copy (the id is the
  // agentserv attachment id once sent).
  const src = local ?? (isImage && isUploaded(attachment.id) ? previewUrl(attachment.id) : undefined);

  return (
    <AttachmentPrimitive.Root
      className={cn(
        'relative flex max-w-56 items-center gap-2 rounded-lg border bg-muted/40 py-1.5 pr-2 pl-1.5 text-xs',
        failed && 'border-destructive/60 text-destructive'
      )}
      title={failed ? 'This file was not uploaded' : attachment.name}
    >
      <div className="flex size-8 shrink-0 items-center justify-center overflow-hidden rounded bg-background">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element -- proxied blob, not an optimizable asset
          <img src={src} alt="" className="size-full object-cover" />
        ) : isImage ? (
          <ImageIcon className="size-4 text-muted-foreground" aria-hidden />
        ) : (
          <FileTextIcon className="size-4 text-muted-foreground" aria-hidden />
        )}
      </div>
      <span className={cn('truncate', failed && 'line-through')}>
        <AttachmentPrimitive.Name />
      </span>
      {uploading && <Loader2Icon className="size-3.5 shrink-0 animate-spin text-muted-foreground" aria-hidden />}
      {removable && (
        <AttachmentPrimitive.Remove
          className="ml-auto shrink-0 rounded-full p-0.5 text-muted-foreground hover:bg-background hover:text-foreground"
          aria-label="Remove attachment"
        >
          <XIcon className="size-3.5" />
        </AttachmentPrimitive.Remove>
      )}
    </AttachmentPrimitive.Root>
  );
};

export const ComposerAttachmentChip: FC = () => <AttachmentChip removable />;
export const MessageAttachmentChip: FC = () => <AttachmentChip />;
