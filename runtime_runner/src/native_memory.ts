import fs from 'fs';
import path from 'path';

import { resolveWorkspaceSessionPath } from './workspace_scope.js';

function envInt(name: string, defaultValue: number): number {
  const raw = (process.env[name] || '').trim();
  if (!raw) return defaultValue;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed)) return defaultValue;
  return parsed;
}

function walkFiles(
  root: string,
  filter: (absPath: string) => boolean,
): string[] {
  const out: string[] = [];
  if (!fs.existsSync(root)) return out;
  const stack: string[] = [root];
  while (stack.length > 0) {
    const current = stack.pop()!;
    let entries: fs.Dirent[] = [];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const ent of entries) {
      const abs = path.join(current, ent.name);
      if (ent.isDirectory()) {
        stack.push(abs);
      } else if (ent.isFile() && filter(abs)) {
        out.push(abs);
      }
    }
  }
  return out;
}

function pathSegments(root: string, absPath: string): string[] {
  return path
    .relative(root, absPath)
    .split(path.sep)
    .filter(Boolean);
}

function isPreferredClaudeSessionFile(root: string, absPath: string): boolean {
  const segments = pathSegments(root, absPath);
  const base = path.basename(absPath).toLowerCase();
  return (
    segments.length >= 2 &&
    segments[0] === 'projects' &&
    path.extname(absPath).toLowerCase() === '.jsonl' &&
    !base.startsWith('agent-')
  );
}

function isFallbackClaudeSessionFile(root: string, absPath: string): boolean {
  const segments = pathSegments(root, absPath).map((s) => s.toLowerCase());
  const ext = path.extname(absPath).toLowerCase();
  const base = path.basename(absPath).toLowerCase();
  if (ext !== '.jsonl') {
    return false;
  }
  if (base.startsWith('agent-')) {
    return false;
  }
  if (segments.some((s) => ['debug', 'todos', 'shell-snapshots', 'skills'].includes(s))) {
    return false;
  }
  return true;
}

function stripClaudeSidechainEntries(raw: string, absPath: string): string {
  if (path.extname(absPath).toLowerCase() !== '.jsonl') {
    return raw;
  }
  const kept: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const parsed = JSON.parse(trimmed) as Record<string, unknown>;
      if (parsed.isSidechain === true) {
        continue;
      }
    } catch {
      // Keep malformed/non-JSON lines rather than dropping transcript content.
    }
    kept.push(line);
  }
  return kept.join('\n');
}

export function collectNativeSessionMemory(workspaceKey: string): string {
  if (!workspaceKey) return '';
  const maxChars = envInt('KCSI_NATIVE_MEMORY_MAX_CHARS', 240_000);
  const maxFiles = envInt('KCSI_NATIVE_MEMORY_MAX_FILES', 8);
  const maxCharsPerFile = envInt(
    'KCSI_NATIVE_MEMORY_MAX_CHARS_PER_FILE',
    60_000,
  );
  if (maxChars <= 0) return '';

  const claudeRoot = path.join(resolveWorkspaceSessionPath(workspaceKey), '.claude');
  const preferredFiles = walkFiles(claudeRoot, (p) =>
    isPreferredClaudeSessionFile(claudeRoot, p),
  );
  const files = preferredFiles.length > 0
    ? preferredFiles
    : walkFiles(claudeRoot, (p) => isFallbackClaudeSessionFile(claudeRoot, p));
  if (files.length === 0) return '';
  files.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);

  const selected = maxFiles > 0 ? files.slice(0, maxFiles) : files;
  const blocks: string[] = [];
  let total = 0;
  for (const file of selected) {
    let raw = '';
    try {
      raw = fs.readFileSync(file, 'utf-8');
    } catch {
      continue;
    }
    raw = stripClaudeSidechainEntries(raw, file);
    if (!raw.trim()) continue;
    const chunk =
      maxCharsPerFile > 0 && raw.length > maxCharsPerFile
        ? raw.slice(-maxCharsPerFile)
        : raw;
    const rel = path.relative(claudeRoot, file);
    const wrapped = `# file: ${rel}\n${chunk}\n`;
    blocks.push(wrapped);
    total += wrapped.length + (blocks.length > 1 ? '\n\n---\n\n'.length : 0);
    if (total >= maxChars) break;
  }

  let merged = blocks.join('\n\n---\n\n').trim();
  if (merged.length > maxChars) {
    merged = merged.slice(-maxChars);
  }
  return merged;
}
