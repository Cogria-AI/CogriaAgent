'use client';

/**
 * Inline propose/confirm card (MVP-A). Rendered in the conversation flow (above
 * the composer), not as a modal. When the agent proposes a write action the SSE
 * stream emits `confirm_required`; ChatThread stashes it and renders this card.
 * The user's choice resolves the promise the chat adapter awaits, which sends
 * the [CONFIRMED]/[CANCELLED] follow-up turn.
 */
import { useState } from 'react';
import { useTranslations } from 'next-intl';

import { Button } from '@/components/ui/button';

export interface PendingConfirmation {
  token: string;
  summary: string;
  actionName?: string;
  resolve: (confirmed: boolean) => void;
}

export default function ConfirmCard({ pending }: { pending: PendingConfirmation }) {
  const t = useTranslations('confirm');
  const [acted, setActed] = useState(false);

  function decide(confirmed: boolean) {
    if (acted) return;
    setActed(true);
    pending.resolve(confirmed);
  }

  return (
    <div
      className="mx-auto w-full max-w-(--thread-max-width) px-2"
      role="group"
      aria-label={t('title')}
    >
      <div className="rounded-lg border border-border bg-muted/40 p-4 text-sm">
        <div className="mb-1 flex items-center gap-2 font-medium text-foreground">
          <span aria-hidden>✋</span>
          {t('title')}
        </div>
        <p className="mb-3 text-foreground">{pending.summary}</p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" disabled={acted} onClick={() => decide(false)}>
            {t('cancel')}
          </Button>
          <Button size="sm" disabled={acted} onClick={() => decide(true)}>
            {t('confirm')}
          </Button>
        </div>
      </div>
    </div>
  );
}
