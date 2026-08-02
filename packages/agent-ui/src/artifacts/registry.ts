/**
 * Artifact tool registry — the front-end's list of "artifact" tools. Unlike
 * action tools (pulled from the backend catalog and executed server-side), an
 * artifact tool is NEVER executed: the LLM calls it purely to push something
 * into the right-hand panel. agentserv sees the call, emits a `tool_call` SSE
 * event, and returns `{}` to the model.
 *
 * These three are framework BUILTINS (domain-neutral). A project may append its
 * own artifact tools (e.g. an iframe preview of a specific site) via
 * `extraArtifactTools` when mounting the chat — see examples/.
 */

export interface ArtifactToolDef {
  name: string;
  description: string;
  params_schema: {
    type: 'object';
    properties: Record<string, { type: string; description?: string }>;
    required?: string[];
  };
}

export const BUILTIN_ARTIFACT_TOOLS: ArtifactToolDef[] = [
  {
    name: 'render_report',
    description:
      'Render a chart or data table in the right-hand panel from data you already ' +
      'fetched via a previous (read) tool call. Call this AFTER you have the numbers, ' +
      'as a second step, when the user wants to SEE a trend / comparison / ranking ' +
      'visually. Pick chart_type: "line" for trends over time, "bar" for category ' +
      'comparisons, "pie" for share/proportion, "table" for detailed rankings (a table ' +
      'gets an Export CSV button). Pass rows in `data`; name the x-axis field via ' +
      '`x_field` and the numeric series via `y_fields`. After calling, just say it is ' +
      'on the right; do not re-read the numbers.',
    params_schema: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Short report title shown above the chart.' },
        chart_type: { type: 'string', description: 'One of "line" | "bar" | "pie" | "table".' },
        data: {
          type: 'array',
          description:
            'Array of row objects, e.g. [{"day":"2026-05-01","value":1200}, ...]. For pie, ' +
            'each row needs a label field (x_field) and one numeric value (y_fields[0]).',
        },
        x_field: { type: 'string', description: 'Field used for the X axis / category / pie label.' },
        y_fields: { type: 'array', description: 'Numeric field name(s) to plot, e.g. ["value"].' },
      },
      required: ['title', 'chart_type', 'data'],
    },
  },
  {
    name: 'preview_image',
    description:
      'Show a single image in the right-hand panel with optional action buttons. Pass ' +
      'the hosted image_url. After calling, just say it is on the right.',
    params_schema: {
      type: 'object',
      properties: {
        image_url: { type: 'string', description: 'Hosted image URL to display.' },
        caption: { type: 'string', description: 'Optional short caption under the image.' },
        prompt: { type: 'string', description: 'The prompt used to generate the image (enables Regenerate).' },
      },
      required: ['image_url'],
    },
  },
  {
    name: 'compare_images',
    description:
      'Show two images side by side in the right-hand panel for the user to pick between. ' +
      'Each side gets a "Choose this" button. Pass both hosted URLs.',
    params_schema: {
      type: 'object',
      properties: {
        left_url: { type: 'string', description: 'Left image URL.' },
        right_url: { type: 'string', description: 'Right image URL.' },
        left_label: { type: 'string', description: 'Optional caption under the left image.' },
        right_label: { type: 'string', description: 'Optional caption under the right image.' },
      },
      required: ['left_url', 'right_url'],
    },
  },
];

/** Resolve the active artifact tool set: builtins + any project extras. */
export function resolveArtifactTools(extra: ArtifactToolDef[] = []): ArtifactToolDef[] {
  return [...BUILTIN_ARTIFACT_TOOLS, ...extra];
}

export function artifactToolNames(defs: ArtifactToolDef[]): ReadonlySet<string> {
  return new Set(defs.map((d) => d.name));
}

export function isArtifactTool(name: string, defs: ArtifactToolDef[]): boolean {
  return defs.some((d) => d.name === name);
}

/**
 * Stable identity for an artifact in the panel store.
 * - Per-instance (one tab per content): images and reports key by content so a
 *   streamed re-render of the SAME call merges into one tab while a genuinely
 *   new call opens its own.
 * - A project's singleton-style tool (one surface, re-invoking updates in place)
 *   can key by name; unknown tools default to name.
 */
export function artifactKeyFor(name: string, args: Record<string, unknown>): string {
  if (name === 'preview_image') {
    const url = typeof args.image_url === 'string' ? args.image_url : '';
    return `preview_image:${url}`;
  }
  if (name === 'compare_images') {
    const l = typeof args.left_url === 'string' ? args.left_url : '';
    const r = typeof args.right_url === 'string' ? args.right_url : '';
    return `compare_images:${l}|${r}`;
  }
  if (name === 'render_report') {
    const title = typeof args.title === 'string' ? args.title : '';
    const ct = typeof args.chart_type === 'string' ? args.chart_type : '';
    return `render_report:${ct}:${title}`;
  }
  return name;
}

/** Key for a "generating…" placeholder driven by a long-running ACTION tool-call
 * (not an artifact tool) whose result later yields an image_url. Keyed by the
 * action's tool-call id so concurrent generations don't collide. */
export function imageGenKeyFor(toolCallId: string): string {
  return `image:gen:${toolCallId}`;
}
