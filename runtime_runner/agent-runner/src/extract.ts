/**
 * Text extraction helpers for agent-runner message processing.
 */

export function extractAssistantText(message: unknown): string | null {
  if (message == null || typeof message !== 'object') return null;
  const msg = message as { message?: { content?: unknown } };
  const content = msg.message?.content;
  if (!Array.isArray(content)) return null;
  const parts = content
    .filter((c) => typeof c === 'object' && c !== null && (c as { type?: string }).type === 'text')
    .map((c) => String((c as { text?: unknown }).text ?? ''))
    .join('')
    .trim();
  return parts || null;
}

export function extractStructuredForumText(message: unknown): string | null {
  const text = extractAssistantText(message);
  if (!text) return null;
  const match = text.match(/(?:^|\n)\s*(INSIGHT|COMMENT)\s*\n[\s\S]*$/m);
  if (!match || match.index === undefined) return null;
  const structured = text.slice(match.index).trim();
  return structured || null;
}
