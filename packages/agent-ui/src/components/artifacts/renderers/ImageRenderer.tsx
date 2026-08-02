'use client';

import { useTranslations } from 'next-intl';

import { Button } from '@/components/ui/button';
import type { RendererProps } from './types';

export default function ImageRenderer({ args, status }: RendererProps) {
  const t = useTranslations('artifacts');
  const url = typeof args.image_url === 'string' ? args.image_url : '';
  const caption = typeof args.caption === 'string' ? args.caption : '';

  if (status === 'generating' || !url) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-4">
        <div className="aspect-square w-full max-w-sm animate-pulse rounded-lg bg-muted" />
        <p className="text-muted-foreground text-sm">{t('generating')}</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url} alt={caption || 'image'} className="w-full rounded-lg object-contain" />
      {caption && <p className="text-muted-foreground text-sm">{caption}</p>}
    </div>
  );
}
