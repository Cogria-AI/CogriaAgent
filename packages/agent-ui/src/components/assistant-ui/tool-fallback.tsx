'use client';

import { memo, useCallback, useRef, useState } from 'react';
import { AlertCircleIcon, CheckIcon, ChevronDownIcon, LoaderIcon, XCircleIcon } from 'lucide-react';
import {
  useScrollLock,
  type ToolCallMessagePartStatus,
  type ToolCallMessagePartComponent,
} from '@assistant-ui/react';

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';

const ANIMATION_DURATION = 200;

type ToolStatus = ToolCallMessagePartStatus['type'];

const statusIconMap: Record<ToolStatus, React.ElementType> = {
  running: LoaderIcon,
  complete: CheckIcon,
  incomplete: XCircleIcon,
  'requires-action': AlertCircleIcon,
};

const ToolFallbackImpl: ToolCallMessagePartComponent = ({ toolName, argsText, result, status }) => {
  const collapsibleRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const lockScroll = useScrollLock(collapsibleRef, ANIMATION_DURATION);
  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) lockScroll();
      setOpen(next);
    },
    [lockScroll]
  );

  const statusType = status?.type ?? 'complete';
  const isRunning = statusType === 'running';
  const Icon = statusIconMap[statusType];

  return (
    <Collapsible
      ref={collapsibleRef}
      open={open}
      onOpenChange={handleOpenChange}
      className="my-1 w-full rounded-lg border py-3"
      style={{ '--animation-duration': `${ANIMATION_DURATION}ms` } as React.CSSProperties}
    >
      <CollapsibleTrigger className="group/trigger flex w-full items-center gap-2 px-4 text-sm">
        <Icon className={cn('size-4 shrink-0', isRunning && 'animate-spin')} />
        <span className="grow text-start leading-none">
          Used tool: <b>{toolName}</b>
        </span>
        <ChevronDownIcon className="size-4 shrink-0 transition-transform group-data-[state=closed]/trigger:-rotate-90" />
      </CollapsibleTrigger>
      <CollapsibleContent className="overflow-hidden text-sm outline-none data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down">
        <div className="mt-3 flex flex-col gap-2 border-t pt-2">
          {argsText && (
            <div className="px-4">
              <pre className="whitespace-pre-wrap">{argsText}</pre>
            </div>
          )}
          {result !== undefined && (
            <div className="border-t border-dashed px-4 pt-2">
              <p className="font-semibold">Result:</p>
              <pre className="whitespace-pre-wrap">
                {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export const ToolFallback = memo(ToolFallbackImpl);
