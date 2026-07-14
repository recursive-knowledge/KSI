export type AnthropicMessage = Record<string, unknown>;

const COMPACT_TOOL_RESULT_MARKER = '[ksi compacted tool result]';

function messageToolResultBlocks(message: AnthropicMessage): Array<Record<string, unknown>> {
  if (message.role !== 'user' || !Array.isArray(message.content)) {
    return [];
  }
  return (message.content as unknown[]).filter((block): block is Record<string, unknown> =>
    Boolean(block && typeof block === 'object' && (block as Record<string, unknown>).type === 'tool_result'),
  );
}

function isAlreadyCompacted(block: Record<string, unknown>): boolean {
  const content = block.content;
  if (!Array.isArray(content) || content.length === 0) {
    return false;
  }
  const first = content[0];
  if (!first || typeof first !== 'object') {
    return false;
  }
  const text = String((first as Record<string, unknown>).text || '');
  return text.includes(COMPACT_TOOL_RESULT_MARKER);
}

export function compactConsumedToolResults(
  messages: AnthropicMessage[],
  toolUseNamesById: Map<string, string>,
  keepRecentToolResultMessages: number,
): void {
  let remainingFullMessages = Math.max(0, keepRecentToolResultMessages);
  for (let i = messages.length - 1; i >= 0; i--) {
    const blocks = messageToolResultBlocks(messages[i]);
    if (blocks.length === 0) {
      continue;
    }
    if (remainingFullMessages > 0) {
      remainingFullMessages -= 1;
      continue;
    }
    for (const block of blocks) {
      if (isAlreadyCompacted(block)) {
        continue;
      }
      const toolUseId = String(block.tool_use_id || '');
      const toolName = toolUseNamesById.get(toolUseId) || 'tool';
      block.content = [{
        type: 'text',
        text:
          `${COMPACT_TOOL_RESULT_MARKER} ${toolName} result was already delivered ` +
          'to the model and has been compacted to reduce scheduled Anthropic input tokens. ' +
          `This placeholder preserves the consumed tool_result turn; it is not an instruction to call ${toolName} again. ` +
          'Do not replay stateful or destructive tools just to recover the omitted payload.',
      }];
    }
  }
}
