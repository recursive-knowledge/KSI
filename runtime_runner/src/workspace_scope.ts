import path from 'path';

import {
  AGENT_WORKSPACES_DIR,
  GLOBAL_WORKSPACE_DIR,
  RUNTIME_IPC_DIR,
  RUNTIME_PROVIDER_SESSIONS_DIR,
  RUNTIME_RUNNER_OVERLAYS_DIR,
  TASK_WORKSPACES_DIR,
} from './config.js';

const KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const TASK_PREFIX = 'task__';
const AGENT_PREFIX = 'agent__';
const RESERVED_KEYS = new Set(['global']);

type WorkspaceKind = 'task' | 'agent';

interface ParsedWorkspaceKey {
  kind: WorkspaceKind;
  key: string;
}

export function isValidWorkspaceKey(workspaceKey: string): boolean {
  if (!workspaceKey) return false;
  if (workspaceKey !== workspaceKey.trim()) return false;
  if (!KEY_PATTERN.test(workspaceKey)) return false;
  if (workspaceKey.includes('/') || workspaceKey.includes('\\')) return false;
  if (workspaceKey.includes('..')) return false;
  if (RESERVED_KEYS.has(workspaceKey.toLowerCase())) return false;
  return workspaceKey.startsWith(TASK_PREFIX) || workspaceKey.startsWith(AGENT_PREFIX);
}

export function assertValidWorkspaceKey(workspaceKey: string): void {
  if (!isValidWorkspaceKey(workspaceKey)) {
    throw new Error(`Invalid workspace key "${workspaceKey}"`);
  }
}

function ensureWithinBase(baseDir: string, resolvedPath: string): void {
  const rel = path.relative(baseDir, resolvedPath);
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error(`Path escapes base directory: ${resolvedPath}`);
  }
}

function parseWorkspaceKey(workspaceKey: string): ParsedWorkspaceKey {
  assertValidWorkspaceKey(workspaceKey);
  if (workspaceKey.startsWith(TASK_PREFIX)) {
    return { kind: 'task', key: workspaceKey.slice(TASK_PREFIX.length) || 'task' };
  }
  return { kind: 'agent', key: workspaceKey.slice(AGENT_PREFIX.length) || 'agent' };
}

function workspaceBaseDir(kind: WorkspaceKind): string {
  return kind === 'agent' ? AGENT_WORKSPACES_DIR : TASK_WORKSPACES_DIR;
}

function runtimeStateBaseDir(root: string, kind: WorkspaceKind): string {
  return path.join(root, kind === 'agent' ? 'agents' : 'tasks');
}

export function resolveWorkspaceRootPath(workspaceKey: string): string {
  const parsed = parseWorkspaceKey(workspaceKey);
  const baseDir = workspaceBaseDir(parsed.kind);
  const resolved = path.resolve(baseDir, parsed.key);
  ensureWithinBase(baseDir, resolved);
  return resolved;
}

export function resolveWorkspaceIpcPath(workspaceKey: string): string {
  const parsed = parseWorkspaceKey(workspaceKey);
  const baseDir = runtimeStateBaseDir(RUNTIME_IPC_DIR, parsed.kind);
  const resolved = path.resolve(baseDir, parsed.key);
  ensureWithinBase(baseDir, resolved);
  return resolved;
}

export function resolveWorkspaceSessionPath(workspaceKey: string): string {
  const parsed = parseWorkspaceKey(workspaceKey);
  const baseDir = runtimeStateBaseDir(RUNTIME_PROVIDER_SESSIONS_DIR, parsed.kind);
  const resolved = path.resolve(baseDir, parsed.key);
  ensureWithinBase(baseDir, resolved);
  return resolved;
}

export function resolveWorkspaceRunnerOverlayPath(workspaceKey: string): string {
  const parsed = parseWorkspaceKey(workspaceKey);
  const baseDir = runtimeStateBaseDir(RUNTIME_RUNNER_OVERLAYS_DIR, parsed.kind);
  const resolved = path.resolve(baseDir, parsed.key);
  ensureWithinBase(baseDir, resolved);
  return resolved;
}

export function resolveGlobalWorkspacePath(): string {
  return GLOBAL_WORKSPACE_DIR;
}
