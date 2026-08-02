'use client';

import { Button } from '@/components/ui/button';
import type { RendererProps } from './types';
import { sendAgentMessage } from './types';

export default function CompareImagesRenderer({ args }: RendererProps) {
  const left = typeof args.left_url === 'string' ? args.left_url : '';
  const right = typeof args.right_url === 'string' ? args.right_url : '';
  const leftLabel = typeof args.left_label === 'string' ? args.left_label : 'A';
  const rightLabel = typeof args.right_label === 'string' ? args.right_label : 'B';

  return (
    <div className="grid h-full grid-cols-2 gap-3 p-4">
      {[
        { url: left, label: leftLabel },
        { url: right, label: rightLabel },
      ].map((side, i) => (
        <div key={i} className="flex flex-col gap-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={side.url} alt={side.label} className="w-full rounded-lg object-contain" />
          <p className="text-center text-muted-foreground text-sm">{side.label}</p>
          <Button variant="outline" size="sm" onClick={() => sendAgentMessage(`Use image: ${side.url}`)}>
            Choose
          </Button>
        </div>
      ))}
    </div>
  );
}
