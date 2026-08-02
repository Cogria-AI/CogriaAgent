'use client';

import { useTranslations } from 'next-intl';
import { ArrowUpIcon, PaperclipIcon, SquareIcon } from 'lucide-react';
import type { FC, ReactNode } from 'react';
import {
  AuiIf,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from '@assistant-ui/react';

import ArtifactCard from '@/components/artifacts/ArtifactCard';
import { MarkdownText } from '@/components/assistant-ui/markdown-text';
import { ToolFallback } from '@/components/assistant-ui/tool-fallback';
import { ComposerAttachmentChip, MessageAttachmentChip } from '@/components/AttachmentChip';
import { Button } from '@/components/ui/button';
import { ATTACHMENTS_ENABLED } from '@/lib/config';
import { cn } from '@/lib/utils';

type ThreadProps = {
  /** Rendered after the message list, above the composer — the inline ConfirmCard. */
  belowMessages?: ReactNode;
};

export const Thread: FC<ThreadProps> = ({ belowMessages }) => {
  return (
    <ThreadPrimitive.Root
      className="@container flex h-full flex-col bg-background"
      style={{ ['--thread-max-width' as string]: '44rem', ['--composer-radius' as string]: '24px' }}
    >
      <ThreadPrimitive.Viewport
        turnAnchor="top"
        className="relative flex flex-1 flex-col overflow-y-scroll scroll-smooth"
      >
        <div className="mx-auto flex w-full max-w-(--thread-max-width) flex-1 flex-col px-4 pt-4">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <ThreadWelcome />
          </AuiIf>

          <div className="mb-10 flex flex-col gap-y-8 empty:hidden">
            <ThreadPrimitive.Messages>{() => <ThreadMessage />}</ThreadPrimitive.Messages>
          </div>

          <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mt-auto flex flex-col gap-4 rounded-t-(--composer-radius) bg-background pb-4 md:pb-6">
            {belowMessages}
            <Composer />
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  if (role === 'user') return <UserMessage />;
  return <AssistantMessage />;
};

const ThreadWelcome: FC = () => {
  const t = useTranslations('chat');
  return (
    <div className="my-auto flex grow flex-col items-center justify-center px-4 text-center">
      <h1 className="font-semibold text-2xl">{t('welcomeTitle')}</h1>
      <p className="text-muted-foreground text-lg">{t('welcomeSubtitle')}</p>
    </div>
  );
};

const Composer: FC = () => {
  const t = useTranslations('chat');
  return (
    <ComposerPrimitive.Root className="relative flex w-full flex-col">
      <div className="flex w-full flex-col gap-2 rounded-(--composer-radius) border bg-background p-2.5 focus-within:ring-2 focus-within:ring-ring/20">
        {ATTACHMENTS_ENABLED && (
          <AuiIf condition={(s) => s.composer.attachments.length > 0}>
            <div className="flex flex-wrap gap-2 px-1 pt-1">
              <ComposerPrimitive.Attachments components={{ Attachment: ComposerAttachmentChip }} />
            </div>
          </AuiIf>
        )}
        <ComposerPrimitive.Input
          placeholder={t('placeholder')}
          className="max-h-32 min-h-10 w-full resize-none bg-transparent px-1.5 py-1 text-sm outline-none placeholder:text-muted-foreground/80"
          rows={1}
          autoFocus
          aria-label={t('placeholder')}
        />
        <div className="flex items-center justify-end gap-1">
          {ATTACHMENTS_ENABLED && (
            <ComposerPrimitive.AddAttachment asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="mr-auto size-8 rounded-full text-muted-foreground"
                aria-label={t('attach')}
                title={t('attach')}
              >
                <PaperclipIcon className="size-4" />
              </Button>
            </ComposerPrimitive.AddAttachment>
          )}
          <AuiIf condition={(s) => !s.thread.isRunning}>
            <ComposerPrimitive.Send asChild>
              <Button type="button" size="icon" className="size-8 rounded-full" aria-label="Send">
                <ArrowUpIcon className="size-4" />
              </Button>
            </ComposerPrimitive.Send>
          </AuiIf>
          <AuiIf condition={(s) => s.thread.isRunning}>
            <ComposerPrimitive.Cancel asChild>
              <Button type="button" size="icon" className="size-8 rounded-full" aria-label="Stop">
                <SquareIcon className="size-3 fill-current" />
              </Button>
            </ComposerPrimitive.Cancel>
          </AuiIf>
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root data-role="assistant" className="relative">
      <div className={cn('wrap-break-word px-2 text-foreground leading-relaxed')}>
        {/* MessagePrimitive.Parts establishes per-part context so MarkdownText +
            tool fallbacks render correctly on assistant-ui 0.14. */}
        <MessagePrimitive.Parts components={{ Text: MarkdownText, tools: { Fallback: ToolFallback } }} />
        <ArtifactCard />
        <MessagePrimitive.Error>
          <ErrorPrimitive.Root className="mt-2 rounded-md border border-destructive bg-destructive/10 p-3 text-destructive text-sm">
            <ErrorPrimitive.Message className="line-clamp-2" />
          </ErrorPrimitive.Root>
        </MessagePrimitive.Error>
      </div>
    </MessagePrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root data-role="user" className="flex flex-col items-end gap-1.5 px-2">
      <AuiIf condition={(s) => (s.message.attachments?.length ?? 0) > 0}>
        <div className="flex max-w-[85%] flex-wrap justify-end gap-2">
          <MessagePrimitive.Attachments components={{ Attachment: MessageAttachmentChip }} />
        </div>
      </AuiIf>
      <div className="wrap-break-word max-w-[85%] rounded-2xl bg-muted px-4 py-2.5 text-foreground empty:hidden">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
};
