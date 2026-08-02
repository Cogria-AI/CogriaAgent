'use client';

import '@assistant-ui/react-markdown/styles/dot.css';

import {
  type CodeHeaderProps,
  MarkdownTextPrimitive,
  unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
  useIsMarkdownCodeBlock,
} from '@assistant-ui/react-markdown';
import remarkGfm from 'remark-gfm';
import { type FC, memo, useState } from 'react';
import { CheckIcon, CopyIcon } from 'lucide-react';

import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button';
import { cn } from '@/lib/utils';

const MarkdownTextImpl = () => {
  return <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} className="aui-md" components={defaultComponents} />;
};

export const MarkdownText = memo(MarkdownTextImpl);

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const { isCopied, copyToClipboard } = useCopyToClipboard();
  const onCopy = () => {
    if (!code || isCopied) return;
    copyToClipboard(code);
  };
  return (
    <div className="mt-2.5 flex items-center justify-between rounded-t-lg border border-border/50 border-b-0 bg-muted/50 px-3 py-1.5 text-xs">
      <span className="font-medium text-muted-foreground lowercase">{language}</span>
      <TooltipIconButton tooltip="Copy" onClick={onCopy}>
        {!isCopied && <CopyIcon />}
        {isCopied && <CheckIcon />}
      </TooltipIconButton>
    </div>
  );
};

function useCopyToClipboard({ copiedDuration = 3000 }: { copiedDuration?: number } = {}) {
  const [isCopied, setIsCopied] = useState(false);
  const copyToClipboard = (value: string) => {
    if (!value || typeof navigator === 'undefined' || !navigator.clipboard) return;
    navigator.clipboard.writeText(value).then(
      () => {
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), copiedDuration);
      },
      () => {}
    );
  };
  return { isCopied, copyToClipboard };
}

const defaultComponents = memoizeMarkdownComponents({
  h1: ({ className, ...props }) => <h1 className={cn('mb-2 font-semibold text-base first:mt-0 last:mb-0', className)} {...props} />,
  h2: ({ className, ...props }) => <h2 className={cn('mt-3 mb-1.5 font-semibold text-sm first:mt-0 last:mb-0', className)} {...props} />,
  h3: ({ className, ...props }) => <h3 className={cn('mt-2.5 mb-1 font-semibold text-sm first:mt-0 last:mb-0', className)} {...props} />,
  p: ({ className, ...props }) => <p className={cn('my-2.5 leading-normal first:mt-0 last:mb-0', className)} {...props} />,
  a: ({ className, ...props }) => <a className={cn('text-primary underline underline-offset-2 hover:text-primary/80', className)} {...props} />,
  blockquote: ({ className, ...props }) => <blockquote className={cn('my-2.5 border-muted-foreground/30 border-s-2 ps-3 text-muted-foreground italic', className)} {...props} />,
  ul: ({ className, ...props }) => <ul className={cn('my-2 ms-4 list-disc [&>li]:mt-1', className)} {...props} />,
  ol: ({ className, ...props }) => <ol className={cn('my-2 ms-4 list-decimal [&>li]:mt-1', className)} {...props} />,
  hr: ({ className, ...props }) => <hr className={cn('my-2 border-muted-foreground/20', className)} {...props} />,
  table: ({ className, ...props }) => <table className={cn('my-2 w-full border-separate border-spacing-0 overflow-y-auto', className)} {...props} />,
  th: ({ className, ...props }) => <th className={cn('bg-muted px-2 py-1 text-start font-medium first:rounded-ss-lg last:rounded-se-lg', className)} {...props} />,
  td: ({ className, ...props }) => <td className={cn('border-muted-foreground/20 border-s border-b px-2 py-1 text-start last:border-e', className)} {...props} />,
  tr: ({ className, ...props }) => <tr className={cn('m-0 border-b p-0 first:border-t', className)} {...props} />,
  li: ({ className, ...props }) => <li className={cn('leading-normal', className)} {...props} />,
  pre: ({ className, ...props }) => <pre className={cn('overflow-x-auto rounded-t-none rounded-b-lg border border-border/50 border-t-0 bg-muted/30 p-3 text-xs leading-relaxed', className)} {...props} />,
  code: function Code({ className, ...props }) {
    const isCodeBlock = useIsMarkdownCodeBlock();
    return (
      <code
        className={cn(!isCodeBlock && 'rounded-md border border-border/50 bg-muted/50 px-1.5 py-0.5 font-mono text-[0.85em]', className)}
        {...props}
      />
    );
  },
  CodeHeader,
});
