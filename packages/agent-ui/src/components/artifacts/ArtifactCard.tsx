'use client';

/**
 * In-flow chip that re-opens the artifact panel when it's collapsed. Rendered at
 * the tail of an assistant message; shows only when there are artifacts and the
 * panel is closed. Generic — no business knowledge.
 */
import { PanelRightOpenIcon } from 'lucide-react';
import { useTranslations } from 'next-intl';

import { Button } from '@/components/ui/button';
import { useArtifactStore } from '@/lib/artifact-store';

export default function ArtifactCard() {
  const t = useTranslations('artifacts');
  const items = useArtifactStore((s) => s.items);
  const open = useArtifactStore((s) => s.open);
  const openPanel = useArtifactStore((s) => s.openPanel);

  if (open || items.length === 0) return null;

  return (
    <Button variant="outline" size="sm" className="mt-2 gap-2" onClick={openPanel}>
      <PanelRightOpenIcon className="size-4" />
      {t('reopen', { count: items.length })}
    </Button>
  );
}
