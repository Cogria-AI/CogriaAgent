/**
 * Artifacts panel state (Zustand) — drives the third pane of the shell.
 *
 * Holds the artifacts the current conversation produced (history capped at
 * MAX_ARTIFACTS, oldest evicted), which one is showing, and whether the panel is
 * open. The chat adapter pushes artifacts here as `tool_call` SSE events arrive;
 * ArtifactPanel renders off it.
 *
 * An artifact tool call is an explicit "show me" intent, so every upsert opens
 * the panel and activates that artifact. Close COLLAPSES (open=false) rather
 * than destroying. Re-invoking the same artifact (same key, see artifactKeyFor)
 * re-opens the same surface instead of spawning a duplicate.
 */
import { create } from 'zustand';

/** Max artifacts retained per conversation. */
export const MAX_ARTIFACTS = 5;

export interface ArtifactItem {
  id: string;
  toolName: string;
  args: Record<string, unknown>;
  createdAt: number;
  /**
   * 'generating' marks a placeholder surface for a long-running ACTION whose
   * result hasn't arrived yet (e.g. an image generator). The renderer shows a
   * skeleton + elapsed timer; once the result arrives the adapter upserts the
   * same id to 'ready' with real args. Undefined means a normal ready artifact.
   */
  status?: 'generating' | 'ready';
}

interface ArtifactState {
  conversationKey: string | null;
  items: ArtifactItem[];
  activeId: string | null;
  open: boolean;
  device: 'desktop' | 'mobile';

  hydrate: (
    conversationKey: string,
    seed: Array<{ id: string; toolName: string; args: Record<string, unknown>; status?: 'generating' | 'ready' }>
  ) => void;
  upsert: (item: {
    id: string;
    toolName: string;
    args: Record<string, unknown>;
    status?: 'generating' | 'ready';
  }) => void;
  setActive: (id: string) => void;
  closePanel: () => void;
  openPanel: () => void;
  setDevice: (device: 'desktop' | 'mobile') => void;
}

export const useArtifactStore = create<ArtifactState>((set, get) => ({
  conversationKey: null,
  items: [],
  activeId: null,
  open: false,
  device: 'desktop',

  hydrate: (conversationKey, seed) => {
    if (get().conversationKey === conversationKey) return;
    const items = seed.slice(-MAX_ARTIFACTS).map((s, i) => ({ ...s, createdAt: Date.now() + i }));
    set({ conversationKey, items, activeId: null, open: false });
  },

  upsert: ({ id, toolName, args, status }) =>
    set((s) => {
      const existing = s.items.find((i) => i.id === id);
      if (existing) {
        return {
          items: s.items.map((i) => (i.id === id ? { ...i, toolName, args, status: status ?? i.status } : i)),
          activeId: id,
          open: true,
        };
      }
      let items = [...s.items, { id, toolName, args, createdAt: Date.now(), status }];
      if (items.length > MAX_ARTIFACTS) items = items.slice(items.length - MAX_ARTIFACTS);
      return { items, activeId: id, open: true };
    }),

  setActive: (id) => set({ activeId: id, open: true }),
  closePanel: () => set({ open: false }),
  openPanel: () => set({ open: true }),
  setDevice: (device) => set({ device }),
}));

/** Resolve the active artifact (falls back to the newest if activeId is stale). */
export function selectActiveArtifact(s: ArtifactState): ArtifactItem | null {
  if (s.items.length === 0) return null;
  return s.items.find((i) => i.id === s.activeId) ?? s.items[s.items.length - 1];
}
