'use client';

import * as React from 'react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

type Props = React.ComponentProps<typeof Button> & {
  tooltip: string;
  side?: 'top' | 'bottom' | 'left' | 'right';
};

/** Minimal icon button with a native title tooltip (no extra tooltip provider). */
export function TooltipIconButton({ tooltip, side: _side, className, children, ...rest }: Props) {
  return (
    <Button
      variant="ghost"
      size="icon"
      title={tooltip}
      aria-label={tooltip}
      className={cn('size-6 p-1', className)}
      {...rest}
    >
      {children}
      <span className="sr-only">{tooltip}</span>
    </Button>
  );
}
