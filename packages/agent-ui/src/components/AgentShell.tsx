'use client';

/**
 * Persistent shell: chat (left) + artifact panel (right). Lives in the route
 * layout so navigating between conversations swaps only the chat content, never
 * the shell. De-businessified: no tenant, no sidebar (agentserv owns
 * persistence in C0; a conversation sidebar can be layered later).
 */
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

import ArtifactPanel from '@/components/artifacts/ArtifactPanel';
import { APP_TITLE } from '@/lib/config';
import { useArtifactStore } from '@/lib/artifact-store';

interface RefreshContext {
  refreshConversations: () => void;
  refreshToken: number;
}

const ConversationRefreshContext = createContext<RefreshContext>({
  refreshConversations: () => {},
  refreshToken: 0,
});

export function useConversationRefresh(): RefreshContext {
  return useContext(ConversationRefreshContext);
}

export default function AgentShell({ children }: { children: ReactNode }) {
  const [refreshToken, setRefreshToken] = useState(0);
  const refreshConversations = useCallback(() => setRefreshToken((n) => n + 1), []);
  const ctx = useMemo(() => ({ refreshConversations, refreshToken }), [refreshConversations, refreshToken]);

  const panelOpen = useArtifactStore((s) => s.open);

  return (
    <ConversationRefreshContext.Provider value={ctx}>
      <div className="flex h-dvh flex-col">
        <header className="flex h-12 shrink-0 items-center border-b px-4 font-medium">{APP_TITLE}</header>
        <div className="flex min-h-0 flex-1">
          <main className="min-w-0 flex-1">{children}</main>
          {panelOpen && (
            <aside className="hidden w-[480px] shrink-0 border-l md:block">
              <ArtifactPanel />
            </aside>
          )}
        </div>
      </div>
    </ConversationRefreshContext.Provider>
  );
}
