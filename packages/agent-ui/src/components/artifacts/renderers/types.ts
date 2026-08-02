export interface RendererProps {
  args: Record<string, unknown>;
  status?: 'generating' | 'ready';
}

/** Bridge an artifact action button into the conversation as a user message.
 * ChatThread listens for this and appends it to the thread. */
export function sendAgentMessage(text: string): void {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent('agent:send', { detail: { text } }));
}
