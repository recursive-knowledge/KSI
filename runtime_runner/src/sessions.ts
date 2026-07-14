import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

import { RUNTIME_SESSION_STATE_DIR } from './config.js';
import {
  resolveWorkspaceIpcPath,
  resolveWorkspaceRootPath,
  resolveWorkspaceRunnerOverlayPath,
  resolveWorkspaceSessionPath,
} from './workspace_scope.js';

import { SessionState } from './types.js';

function ensureDir(p: string): void {
  fs.mkdirSync(p, { recursive: true });
}

function writeText(filePath: string, content: string): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content, 'utf-8');
}

function readJsonFile<T>(filePath: string): T {
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw) as T;
}

function removePathIfExists(targetPath: string): void {
  if (!fs.existsSync(targetPath)) {
    return;
  }
  try {
    fs.rmSync(targetPath, {
      recursive: true,
      force: true,
      maxRetries: 8,
      retryDelay: 50,
    });
    return;
  } catch {
    // Fall through to child-by-child cleanup for occasional ENOTEMPTY races.
  }

  try {
    const stat = fs.lstatSync(targetPath);
    if (!stat.isDirectory()) {
      fs.rmSync(targetPath, { force: true });
      return;
    }
    for (const entry of fs.readdirSync(targetPath)) {
      removePathIfExists(path.join(targetPath, entry));
    }
    fs.rmdirSync(targetPath);
  } catch {
    // best-effort
  }
}

export function sessionStateRoot(): string {
  const configured = (process.env.KSI_SESSION_STATE_ROOT || '').trim();
  if (configured) {
    return path.resolve(configured);
  }
  return path.resolve(RUNTIME_SESSION_STATE_DIR);
}

function sessionStateDirForAgent(agentId: string): string {
  if (typeof agentId !== 'string' || agentId.trim() === '') {
    throw new Error(
      `Unsafe agent_id for session state path: ${String(agentId)}`
    );
  }
  const digest = crypto.createHash('sha256').update(agentId, 'utf8').digest('hex');
  return `agent-${digest}`;
}

export function sessionStatePath(agentId: string): string {
  const root = path.resolve(sessionStateRoot());
  return path.join(root, sessionStateDirForAgent(agentId), '.ksi_session.json');
}

export function loadSessionForAgent(agentId: string): string | undefined {
  let p: string;
  try {
    p = sessionStatePath(agentId);
  } catch {
    return undefined;
  }
  if (!fs.existsSync(p)) {
    return undefined;
  }
  try {
    const data = readJsonFile<SessionState>(p);
    // readJsonFile casts blindly; a partially-written or malformed session file
    // (e.g. {"session_id": null} after a crash) must not be returned as a bogus
    // id. Only accept a non-empty string.
    return typeof data?.session_id === 'string' && data.session_id ? data.session_id : undefined;
  } catch {
    return undefined;
  }
}

export function saveSessionForAgent(agentId: string, sessionId: string): void {
  const payload = JSON.stringify({ session_id: sessionId }, null, 2) + '\n';
  try {
    writeText(sessionStatePath(agentId), payload);
  } catch {
    // ignore invalid agent IDs in production; session state is best-effort.
  }
}

export function cleanupWorkspace(workspaceKey: string): void {
  const dirs = [
    resolveWorkspaceRootPath(workspaceKey),
    resolveWorkspaceSessionPath(workspaceKey),
    resolveWorkspaceRunnerOverlayPath(workspaceKey),
    resolveWorkspaceIpcPath(workspaceKey),
  ];
  for (const dir of dirs) {
    removePathIfExists(dir);
  }
}
