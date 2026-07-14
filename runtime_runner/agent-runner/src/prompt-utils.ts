export function buildSystemPromptAppend(
  globalClaudeMd: string | undefined,
  seedContext: string | undefined,
): string | undefined {
  const blocks: string[] = [];
  if (typeof globalClaudeMd === 'string' && globalClaudeMd.trim()) {
    blocks.push(globalClaudeMd.trim());
  }
  if (typeof seedContext === 'string' && seedContext.trim()) {
    blocks.push(['# Ephemeral Seed Context', '', seedContext.trim()].join('\n'));
  }
  return blocks.length > 0 ? blocks.join('\n\n') : undefined;
}

export function summarizeToolInput(
  toolName: string,
  input: unknown,
): Record<string, unknown> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    return {};
  }
  const obj = input as Record<string, unknown>;
  const summary: Record<string, unknown> = {};
  for (const key of ['file_path', 'path', 'old_path', 'new_path', 'pattern']) {
    const value = obj[key];
    if (typeof value === 'string' && value.trim()) {
      summary[key] = value;
    }
  }
  if (
    toolName === 'Read' &&
    typeof obj.file_path === 'string' &&
    obj.file_path.trim()
  ) {
    summary.read_file_path = obj.file_path;
  }
  return summary;
}
