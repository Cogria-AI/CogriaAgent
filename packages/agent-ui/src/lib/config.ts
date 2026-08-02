/**
 * Front-end + BFF configuration, all env-driven: hosts, locale set and branding
 * are values here rather than constants scattered through the code, so the same
 * build serves any deployment.
 *
 * No tenant dimension (single-tenant by design).
 */

function env(name: string, fallback = ''): string {
  return (process.env[name] || '').trim() || fallback;
}

/** Same, for NEXT_PUBLIC_* values that must survive into the browser bundle.
 *
 * The bundler only substitutes `process.env.NEXT_PUBLIC_X` written as a literal
 * member access; a computed `process.env[name]` is left alone and reads as
 * undefined client-side. Passing the already-inlined value in keeps the literal
 * at the call site, so server and client agree — otherwise every public knob
 * silently falls back to its default in the browser and hydration mismatches. */
function publicEnv(value: string | undefined, fallback = ''): string {
  return (value || '').trim() || fallback;
}

/** Where the BFF forwards chat to (the private agentserv). */
export const AGENTSERV_URL = env('AGENTSERV_URL', 'http://127.0.0.1:8001').replace(/\/$/, '');

/** Full URL the BFF calls to exchange the session cookie for a JWT. Explicit —
 * no hostname sniffing. e.g. https://api.example.com/agent-auth/exchange */
export const EXCHANGE_ENDPOINT = env('EXCHANGE_ENDPOINT').replace(/\/$/, '');

/** Optional base for conversation persistence proxied by the BFF. Empty when
 * agentserv owns persistence (the default). */
export const CONVERSATION_BASE = env('CONVERSATION_BASE').replace(/\/$/, '');

export const REDIS_URL = env('REDIS_URL', 'redis://127.0.0.1:6379/2');
export const JWT_CACHE_TTL_SECONDS = Number(env('JWT_CACHE_TTL_SECONDS', '1500'));

/** Supported UI locales (comma-separated env), first is the default. */
export const SUPPORTED_LOCALES: readonly string[] = env('SUPPORTED_LOCALES', 'en')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
export const DEFAULT_LOCALE = SUPPORTED_LOCALES[0] || 'en';

/** Branding. */
export const APP_TITLE = publicEnv(process.env.NEXT_PUBLIC_APP_TITLE, 'Cogria Agent');
export const APP_DESCRIPTION = publicEnv(
  process.env.NEXT_PUBLIC_APP_DESCRIPTION,
  'A conversational agent for your application.'
);

/** File uploads. Off unless the server has attachments wired (agentserv reports
 * it at /health) — this flag only controls whether the UI offers the button. */
export const ATTACHMENTS_ENABLED = publicEnv(process.env.NEXT_PUBLIC_ATTACHMENTS_ENABLED) === '1';

/** BFF-side size gate; keep it at or below agentserv's own max_file_bytes. */
export const MAX_ATTACHMENT_BYTES = Number(env('MAX_ATTACHMENT_BYTES', String(20 * 1024 * 1024)));

/** Same limit, readable in the browser: files are only uploaded when the
 * message is sent, so the composer checks the size at pick time to refuse an
 * oversized file immediately rather than at send. */
export const MAX_ATTACHMENT_BYTES_CLIENT = Number(
  publicEnv(process.env.NEXT_PUBLIC_MAX_ATTACHMENT_BYTES, String(20 * 1024 * 1024))
);

/** File picker filter. Match it to the server's allowed types — in particular,
 * drop image/* when no AGENT_VISION_MODEL is configured, or the user can pick an
 * image the server will refuse. Empty = the built-in default list. */
export const ATTACHMENTS_ACCEPT = publicEnv(process.env.NEXT_PUBLIC_ATTACHMENTS_ACCEPT) || undefined;

/** Optional: a long-running action whose result yields an image_url, for which
 * the chat adapter paints a "generating…" placeholder (e.g. an image generator).
 * Empty disables the placeholder behaviour. */
export const LONG_RUNNING_IMAGE_ACTION = publicEnv(
  process.env.NEXT_PUBLIC_LONG_RUNNING_IMAGE_ACTION
);
