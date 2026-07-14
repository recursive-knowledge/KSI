/**
 * Tool-trace correlation helpers for the Claude Agent SDK message stream.
 *
 * The SDK emits a tool_call (assistant `tool_use` block, no output) and its
 * tool_result as SEPARATE messages. These helpers record the calls and
 * backfill outputs onto them by `tool_use_id`, so downstream scorers (e.g.
 * the ARC fast-path) see complete tool entries instead of falling back to
 * fragile text parsing.
 */
import { summarizeToolInput } from './prompt-utils.js';

/**
 * Extract `tool_use` blocks from an assistant message: record the tool names
 * on the per-message trace entry and push a `tool_call` entry per tool,
 * registering each in `pendingToolCallsById` for later tool_result backfill.
 * No-op for non-assistant messages. Mutates `record`, `toolTrace`, and
 * `pendingToolCallsById` in place.
 */
export function recordAssistantToolUses(
  message: unknown,
  messageCount: number,
  record: Record<string, unknown>,
  toolTrace: Array<Record<string, unknown>>,
  pendingToolCallsById: Map<string, Record<string, unknown>>,
): void {
  if ((message as { type?: string }).type !== 'assistant') return;
  const msgContent = (
    message as {
      message?: {
        content?: Array<{
          type: string;
          name?: string;
          id?: string;
          input?: unknown;
        }>;
      };
    }
  ).message?.content;
  if (!Array.isArray(msgContent)) return;
  const toolUses = msgContent
    .filter((block) => block.type === 'tool_use' && block.name)
    .map((block) => ({
      name: block.name!,
      id: block.id,
      input: block.input,
    }));
  if (toolUses.length === 0) return;
  record.tool_uses = toolUses.map((t) => t.name);
  // Also emit individual tool trace entries for each tool call
  for (const tu of toolUses) {
    const callEntry: Record<string, unknown> = {
      idx: messageCount,
      type: 'tool_call',
      tool_name: tu.name,
      tool_use_id: tu.id,
      ...summarizeToolInput(tu.name, tu.input),
      ts: record.ts,
    };
    toolTrace.push(callEntry);
    if (typeof tu.id === 'string' && tu.id) {
      pendingToolCallsById.set(tu.id, callEntry);
    }
  }
}

/**
 * Backfill `tool_output` onto prior `tool_call` trace entries when a user
 * message arrives carrying `tool_result` blocks, paired by `tool_use_id`.
 * No-op for non-user messages. Mutates the entries in `pendingToolCallsById`
 * (and removes paired ids) in place.
 */
export function backfillToolResults(
  message: unknown,
  pendingToolCallsById: Map<string, Record<string, unknown>>,
  maxChars: number,
): void {
  if ((message as { type?: string }).type !== 'user') return;
  const userContent = (
    message as {
      message?: {
        content?:
          | string
          | Array<{
              type?: string;
              tool_use_id?: string;
              content?: unknown;
              is_error?: boolean;
            }>;
      };
    }
  ).message?.content;
  if (!Array.isArray(userContent)) return;
  for (const block of userContent) {
    if (!block || block.type !== 'tool_result') continue;
    const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
    if (!id) continue;
    const callEntry = pendingToolCallsById.get(id);
    if (!callEntry) continue;
    let outputText: string;
    try {
      if (typeof block.content === 'string') {
        outputText = block.content;
      } else if (Array.isArray(block.content)) {
        // Anthropic tool_result.content is typically [{type:'text', text:'...'}, ...].
        // Concatenate text blocks so downstream JSON.parse sees the raw
        // payload, not a wrapping array.
        const parts: string[] = [];
        let nonTextSeen = false;
        for (const c of block.content as Array<{ type?: string; text?: string }>) {
          if (c && typeof c === 'object' && c.type === 'text' && typeof c.text === 'string') {
            parts.push(c.text);
          } else {
            nonTextSeen = true;
          }
        }
        outputText = nonTextSeen && parts.length === 0
          ? JSON.stringify(block.content)
          : parts.join('');
      } else {
        outputText = JSON.stringify(block.content);
      }
    } catch {
      outputText = '[unserializable_tool_output]';
    }
    if (outputText && outputText.length > maxChars) {
      outputText = outputText.slice(0, maxChars);
      callEntry._truncated = true;
    }
    callEntry.tool_output = outputText;
    if (block.is_error) callEntry.tool_is_error = true;
    pendingToolCallsById.delete(id);
  }
}
