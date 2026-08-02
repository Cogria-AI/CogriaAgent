'use client';

/**
 * Right-hand artifact panel. Reads the artifact store, shows a tab per artifact
 * (capped), and dispatches the active one to a renderer by toolName. Builtin
 * renderers: render_report / preview_image / compare_images. Unknown tools fall
 * back to a JSON dump.
 */
import { XIcon } from 'lucide-react';

import CompareImagesRenderer from '@/components/artifacts/renderers/CompareImagesRenderer';
import ImageRenderer from '@/components/artifacts/renderers/ImageRenderer';
import ReportRenderer from '@/components/artifacts/renderers/ReportRenderer';
import type { RendererProps } from '@/components/artifacts/renderers/types';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { selectActiveArtifact, useArtifactStore } from '@/lib/artifact-store';

const RENDERERS: Record<string, (p: RendererProps) => React.ReactNode> = {
  render_report: ReportRenderer,
  preview_image: ImageRenderer,
  compare_images: CompareImagesRenderer,
};

export default function ArtifactPanel() {
  const items = useArtifactStore((s) => s.items);
  const active = useArtifactStore(selectActiveArtifact);
  const activeId = useArtifactStore((s) => s.activeId);
  const setActive = useArtifactStore((s) => s.setActive);
  const closePanel = useArtifactStore((s) => s.closePanel);

  if (!active) return null;
  const Renderer = RENDERERS[active.toolName];

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-12 shrink-0 items-center gap-2 border-b px-2">
        <div className="flex min-w-0 flex-1 gap-1 overflow-x-auto">
          {items.map((it) => (
            <button
              key={it.id}
              onClick={() => setActive(it.id)}
              className={cn(
                'shrink-0 rounded-md px-2 py-1 text-xs',
                it.id === (activeId ?? active.id) ? 'bg-muted font-medium' : 'text-muted-foreground hover:bg-muted/50'
              )}
            >
              {it.toolName}
            </button>
          ))}
        </div>
        <Button variant="ghost" size="icon" className="size-8 shrink-0" aria-label="Close" onClick={closePanel}>
          <XIcon className="size-4" />
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {Renderer ? (
          <Renderer args={active.args} status={active.status} />
        ) : (
          <pre className="overflow-auto p-4 text-xs">{JSON.stringify(active.args, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
