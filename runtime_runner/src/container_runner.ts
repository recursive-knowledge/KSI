/**
 * Container runner for the shared container runtime.
 * Spawns agent execution in containers and handles IPC.
 *
 * Mount construction lives in ./container_mounts.ts, the `docker run`
 * argument/secret building in ./container_args.ts, and the stdout-envelope
 * parsing in ./container_output.ts. This module owns the spawn + stream +
 * timeout + completion orchestration in {@link runContainerAgent}.
 */
import { ChildProcess, execFile, spawn } from 'child_process';
import { randomBytes } from 'crypto';
import fs from 'fs';
import path from 'path';

import {
  CONTAINER_MAX_OUTPUT_SIZE,
  CONTAINER_TIMEOUT,
  IDLE_TIMEOUT,
} from './config.js';
import {
  resolveWorkspaceIpcPath,
  resolveWorkspaceRootPath,
} from './workspace_scope.js';
import { logger } from './logger.js';
import { CONTAINER_RUNTIME_BIN, stopContainer } from './container_runtime.js';
import { RegisteredWorkspace } from './container_types.js';
import { ContainerInput, ContainerOutput } from './shared_types.js';
import {
  OUTPUT_START_MARKER,
  OUTPUT_END_MARKER,
  isContainerOutput,
  outputHasExpectedNonce,
  parseContainerOutputBlock,
  extractLastContainerOutput,
} from './container_output.js';
import { appendMemoryAndArcMounts, buildVolumeMounts } from './container_mounts.js';
import { buildContainerArgs, ensureEgressInfra, readSecrets } from './container_args.js';

export type { ContainerInput, ContainerOutput };

export const STREAM_PARSE_BUFFER_MAX_CHARS = 2_000_000;

export function trimStreamParseBuffer(
  buffer: string,
  maxChars = STREAM_PARSE_BUFFER_MAX_CHARS,
): string {
  if (buffer.length <= maxChars) return buffer;
  const safeMax = Math.max(maxChars, OUTPUT_START_MARKER.length);
  const startIdx = buffer.indexOf(OUTPUT_START_MARKER);
  if (startIdx === -1) {
    return buffer.slice(-Math.min(safeMax, OUTPUT_START_MARKER.length - 1));
  }

  const fromStart = buffer.slice(startIdx);
  if (fromStart.length <= safeMax) return fromStart;

  const laterStartIdx = fromStart.indexOf(OUTPUT_START_MARKER, OUTPUT_START_MARKER.length);
  if (laterStartIdx !== -1) {
    return trimStreamParseBuffer(fromStart.slice(laterStartIdx), safeMax);
  }

  const keepTail = safeMax - OUTPUT_START_MARKER.length;
  return OUTPUT_START_MARKER + fromStart.slice(-keepTail);
}

export async function runContainerAgent(
  workspaceRuntime: RegisteredWorkspace,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const startTime = Date.now();

  const workspaceRoot = resolveWorkspaceRootPath(workspaceRuntime.folder);
  fs.mkdirSync(workspaceRoot, { recursive: true });

  const mounts = buildVolumeMounts(workspaceRuntime);

  // Mount the knowledge DB / MCP server (and ARC snapshot under --no-memory)
  // when agent-facing memory or ARC tools are configured.
  appendMemoryAndArcMounts(mounts, input, workspaceRuntime);

  const safeName = workspaceRuntime.folder.replace(/[^a-zA-Z0-9-]/g, '-');
  const containerName = `kcsi-runtime-${safeName}-${Date.now()}`;
  const egress = ensureEgressInfra();
  const containerArgs = buildContainerArgs(mounts, containerName, egress);

  logger.debug(
    {
      group: workspaceRuntime.name,
      containerName,
      mounts: mounts.map(
        (m) =>
          `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
      ),
      containerArgs: containerArgs.join(' '),
    },
    'Container mount configuration',
  );

  logger.info(
    {
      group: workspaceRuntime.name,
      containerName,
      mountCount: mounts.length,
    },
    'Spawning container agent',
  );

  const logsDir = path.join(workspaceRoot, 'logs');
  fs.mkdirSync(logsDir, { recursive: true });

  return new Promise((resolve) => {
    const container = spawn(CONTAINER_RUNTIME_BIN, containerArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    onProcess(container, containerName);

    let stdout = '';
    let stderr = '';
    let stdoutTruncated = false;
    let stderrTruncated = false;

    const protocolNonce = randomBytes(16).toString('hex');
    input.protocolNonce = protocolNonce;

    // Pass secrets via stdin (never written to disk or mounted as files)
    input.secrets = readSecrets();
    container.stdin.write(JSON.stringify(input));
    container.stdin.end();
    // Remove secrets from input so they don't appear in logs
    delete input.secrets;

    // Streaming output: parse OUTPUT_START/END marker pairs as they arrive
    let parseBuffer = '';
    let parseBufferTrimmed = false;
    let newSessionId: string | undefined;
    let outputChain = Promise.resolve();

    container.stdout.on('data', (data) => {
      const chunk = data.toString();

      // Always accumulate for logging
      if (!stdoutTruncated) {
        const remaining = CONTAINER_MAX_OUTPUT_SIZE - stdout.length;
        if (chunk.length > remaining) {
          stdout += chunk.slice(0, remaining);
          stdoutTruncated = true;
          logger.warn(
            { group: workspaceRuntime.name, size: stdout.length },
            'Container stdout truncated due to size limit',
          );
        } else {
          stdout += chunk;
        }
      }

      // Stream-parse for output markers
      if (onOutput) {
        parseBuffer += chunk;
        let startIdx: number;
        while ((startIdx = parseBuffer.indexOf(OUTPUT_START_MARKER)) !== -1) {
          const endIdx = parseBuffer.indexOf(OUTPUT_END_MARKER, startIdx);
          if (endIdx === -1) break; // Incomplete pair, wait for more data

          const jsonStr = parseBuffer
            .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
            .trim();
          parseBuffer = parseBuffer.slice(endIdx + OUTPUT_END_MARKER.length);

          try {
            const parsed = parseContainerOutputBlock(jsonStr, protocolNonce);
            if (!parsed) {
              throw new Error(
                'No parseable JSON with expected protocol nonce found inside output marker block',
              );
            }
            logger.debug(
              {
                group: workspaceRuntime.name,
                containerName,
                status: parsed.status,
                resultLength:
                  typeof parsed.result === 'string' ? parsed.result.length : null,
                hasSession: Boolean(parsed.newSessionId),
              },
              'Parsed container output marker',
            );
            if (parsed.newSessionId) {
              newSessionId = parsed.newSessionId;
            }
            hadStreamingOutput = true;
            // Activity detected — reset the hard timeout
            resetTimeout();
            // Call onOutput for all markers (including null results)
            // so idle timers start even for "silent" query completions.
            outputChain = outputChain.then(() => onOutput(parsed));
            // Bench/scheduled runs are single-task executions: request clean
            // container shutdown right after first emitted output marker.
            try {
              const closePath = path.join(
                resolveWorkspaceIpcPath(workspaceRuntime.folder),
                'input',
                '_close',
              );
              fs.writeFileSync(closePath, '');
            } catch {
              // best-effort signal
            }
          } catch (err) {
            logger.warn(
              {
                group: workspaceRuntime.name,
                error: err,
                blockPreview: jsonStr.slice(0, 400),
              },
              'Failed to parse streamed output chunk',
            );
          }
        }
        const beforeTrimLength = parseBuffer.length;
        parseBuffer = trimStreamParseBuffer(parseBuffer);
        if (parseBuffer.length < beforeTrimLength && !parseBufferTrimmed) {
          parseBufferTrimmed = true;
          logger.warn(
            { group: workspaceRuntime.name, size: beforeTrimLength, retained: parseBuffer.length },
            'Container stream parse buffer trimmed due to size limit',
          );
        }
      }
    });

    container.stderr.on('data', (data) => {
      const chunk = data.toString();
      const lines = chunk.trim().split('\n');
      for (const line of lines) {
        if (line) logger.debug({ container: workspaceRuntime.folder }, line);
      }
      // Activity detected on stderr — reset the timeout so active work
      // (tool calls, test runs) extends the container's life, capped by
      // the absolute deadline set at container start.
      resetTimeout();
      if (stderrTruncated) return;
      const remaining = CONTAINER_MAX_OUTPUT_SIZE - stderr.length;
      if (chunk.length > remaining) {
        stderr += chunk.slice(0, remaining);
        stderrTruncated = true;
        logger.warn(
          { group: workspaceRuntime.name, size: stderr.length },
          'Container stderr truncated due to size limit',
        );
      } else {
        stderr += chunk;
      }
    });

    let timedOut = false;
    let hadStreamingOutput = false;
    // Grace period: hard timeout must be at least IDLE_TIMEOUT + 30s so the
    // graceful _close sentinel has time to trigger before the hard kill fires.
    // CONTAINER_TIMEOUT <= 0 explicitly disables the hard deadline; TB2
    // fairness runs rely on task.toml's per-task timeout instead.
    const hardTimeoutEnabled = CONTAINER_TIMEOUT > 0;
    const timeoutMs = hardTimeoutEnabled
      ? Math.max(CONTAINER_TIMEOUT, IDLE_TIMEOUT + 30_000)
      : 0;

    // Absolute deadline: no reset can extend beyond this point.
    const absoluteDeadline = hardTimeoutEnabled ? Date.now() + timeoutMs : 0;

    const killOnTimeout = () => {
      timedOut = true;
      logger.error(
        { group: workspaceRuntime.name, containerName },
        'Container timeout, stopping gracefully',
      );
      execFile(CONTAINER_RUNTIME_BIN, stopContainer(containerName), { timeout: 15000 }, (err: Error | null) => {
        if (err) {
          logger.warn(
            { group: workspaceRuntime.name, containerName, err },
            'Graceful stop failed, force killing',
          );
          container.kill('SIGKILL');
        }
      });
    };

    let timeout: ReturnType<typeof setTimeout> | null = hardTimeoutEnabled
      ? setTimeout(killOnTimeout, timeoutMs)
      : null;

    // Reset the timeout whenever there's activity (streaming output or stderr),
    // but never extend beyond the absolute deadline.
    const resetTimeout = () => {
      if (!hardTimeoutEnabled) return;
      if (timeout) clearTimeout(timeout);
      const remaining = absoluteDeadline - Date.now();
      if (remaining <= 0) {
        killOnTimeout();
        return;
      }
      timeout = setTimeout(killOnTimeout, Math.min(timeoutMs, remaining));
    };

    container.on('close', (code) => {
      if (timeout) clearTimeout(timeout);
      const duration = Date.now() - startTime;

      if (timedOut) {
        const ts = new Date().toISOString().replace(/[:.]/g, '-');
        const timeoutLog = path.join(logsDir, `container-${ts}.log`);
        fs.writeFileSync(
          timeoutLog,
          [
            `=== Container Run Log (TIMEOUT) ===`,
            `Timestamp: ${new Date().toISOString()}`,
            `Workspace: ${workspaceRuntime.name}`,
            `Container: ${containerName}`,
            `Duration: ${duration}ms`,
            `Exit Code: ${code}`,
            `Had Streaming Output: ${hadStreamingOutput}`,
          ].join('\n'),
        );

        // Timeout after output = idle cleanup, not failure.
        // The agent already sent its response; this is just the
        // container being reaped after the idle period expired.
        if (hadStreamingOutput) {
          logger.info(
            { group: workspaceRuntime.name, containerName, duration, code },
            'Container timed out after output (idle cleanup)',
          );
          outputChain
            .then(() => {
              resolve({
                status: 'success',
                result: null,
                newSessionId,
              });
            })
            .catch((err) => {
              logger.error({ group: workspaceRuntime.name, containerName, err }, 'outputChain rejected');
              resolve({ status: 'error', result: null, error: String(err) });
            });
          return;
        }

        logger.error(
          { group: workspaceRuntime.name, containerName, duration, code },
          'Container timed out with no output',
        );

        resolve({
          status: 'error',
          result: null,
          error: `Container timed out after ${CONTAINER_TIMEOUT}ms`,
        });
        return;
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const logFile = path.join(logsDir, `container-${timestamp}.log`);
      const isVerbose =
        process.env.LOG_LEVEL === 'debug' || process.env.LOG_LEVEL === 'trace';

      const logLines = [
        `=== Container Run Log ===`,
        `Timestamp: ${new Date().toISOString()}`,
        `Workspace: ${workspaceRuntime.name}`,
        `Duration: ${duration}ms`,
        `Exit Code: ${code}`,
        `Stdout Truncated: ${stdoutTruncated}`,
        `Stderr Truncated: ${stderrTruncated}`,
        ``,
      ];

      // Treat agent-runner error envelopes (`"status":"error"` on stdout) as
      // errors for diagnostic logging too — without this, the agent-runner
      // catches an SDK throw, emits an error envelope, and exits 0, so the
      // container log path skips the stderr/args/mounts dump and the actual
      // failure reason is invisible. See haiku silent_exit on polyglot/swebench
      // (PR #553) — root-caused via the stderr-tee patch but the captured
      // stderr was blackholed by this gate.
      const isError = code !== 0 || stdout.includes('"status":"error"');

      if (isVerbose || isError) {
        logLines.push(
          `=== Input ===`,
          JSON.stringify(input, null, 2),
          ``,
          `=== Container Args ===`,
          containerArgs.join(' '),
          ``,
          `=== Mounts ===`,
          mounts
            .map(
              (m) =>
                `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
            )
            .join('\n'),
          ``,
          `=== Stderr${stderrTruncated ? ' (TRUNCATED)' : ''} ===`,
          stderr,
          ``,
          `=== Stdout${stdoutTruncated ? ' (TRUNCATED)' : ''} ===`,
          stdout,
        );
      } else {
        logLines.push(
          `=== Input Summary ===`,
          `Prompt length: ${input.prompt.length} chars`,
          `Session ID: ${input.sessionId || 'new'}`,
          ``,
          `=== Mounts ===`,
          mounts
            .map((m) => `${m.containerPath}${m.readonly ? ' (ro)' : ''}`)
            .join('\n'),
          ``,
        );
      }

      fs.writeFileSync(logFile, logLines.join('\n'));
      logger.debug({ logFile, verbose: isVerbose }, 'Container log written');

      if (code !== 0) {
        logger.error(
          {
            group: workspaceRuntime.name,
            code,
            duration,
            stderr,
            stdout,
            logFile,
          },
          'Container exited with error',
        );

        resolve({
          status: 'error',
          result: null,
          error: `Container exited with code ${code}: ${stderr.slice(-200)}`,
        });
        return;
      }

      // Streaming mode: wait for output chain to settle, return completion marker
      if (onOutput) {
        const salvageOutput =
          !hadStreamingOutput ? extractLastContainerOutput(stdout, protocolNonce) : null;
        if (salvageOutput) {
          logger.debug(
            {
              group: workspaceRuntime.name,
              containerName,
              status: salvageOutput.status,
              resultLength:
                typeof salvageOutput.result === 'string'
                  ? salvageOutput.result.length
                  : null,
            },
            'Recovered container output from accumulated stdout',
          );
          outputChain = outputChain.then(() => onOutput(salvageOutput));
        }
        // Silent-failure guard: if the container exited cleanly but we never
        // saw an OUTPUT_START marker and couldn't salvage any JSON from stdout,
        // the agent-runner produced nothing observable. Surface this as an
        // error instead of "success with result=null" so the orchestrator
        // can mark the attempt as failed. Previously this path returned
        // status='success' and the empty output propagated downstream as
        // arc_session parse_error:empty_model_output with no diagnostic trail.
        const observedAnyOutput = hadStreamingOutput || salvageOutput != null;
        outputChain
          .then(() => {
            if (!observedAnyOutput) {
              logger.error(
                {
                  group: workspaceRuntime.name,
                  duration,
                  containerName,
                  stderrTail: stderr.slice(-500),
                  stdoutLen: stdout.length,
                },
                'Container completed with no observable output (silent failure)',
              );
              resolve({
                status: 'error',
                result: null,
                newSessionId,
                error:
                  'agent-runner produced no output: container exited cleanly but ' +
                  'emitted no OUTPUT_START marker and no salvageable JSON on stdout. ' +
                  'Likely a silent claude-agent-sdk startup failure. stderr_tail=' +
                  JSON.stringify(stderr.slice(-400)),
              });
              return;
            }
            logger.info(
              { group: workspaceRuntime.name, duration, newSessionId },
              'Container completed (streaming mode)',
            );
            resolve({
              status: 'success',
              result: null,
              newSessionId,
            });
          })
          .catch((err) => {
            logger.error({ group: workspaceRuntime.name, containerName, err }, 'outputChain rejected');
            resolve({ status: 'error', result: null, error: String(err) });
          });
        return;
      }

      // Legacy mode: parse the last output marker pair from accumulated stdout
      try {
        // Extract JSON between sentinel markers for robust parsing
        const startIdx = stdout.indexOf(OUTPUT_START_MARKER);
        const endIdx = stdout.indexOf(OUTPUT_END_MARKER);

        let jsonLine: string;
        if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
          jsonLine = stdout
            .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
            .trim();
        } else {
          // Fallback: last non-empty line (backwards compatibility)
          const lines = stdout.trim().split('\n');
          jsonLine = lines[lines.length - 1];
        }

        const parsedOutput: unknown = JSON.parse(jsonLine);
        if (!isContainerOutput(parsedOutput)) {
          throw new Error(
            'Container output is not a valid ContainerOutput envelope (missing or invalid "status")',
          );
        }
        const output: ContainerOutput = parsedOutput;
        if (!outputHasExpectedNonce(output, protocolNonce)) {
          throw new Error('Container output missing expected protocol nonce');
        }

        logger.info(
          {
            group: workspaceRuntime.name,
            duration,
            status: output.status,
            hasResult: !!output.result,
          },
          'Container completed',
        );

        resolve(output);
      } catch (err) {
        logger.error(
          {
            group: workspaceRuntime.name,
            stdout,
            stderr,
            error: err,
          },
          'Failed to parse container output',
        );

        resolve({
          status: 'error',
          result: null,
          error: `Failed to parse container output: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
    });

    container.on('error', (err) => {
      if (timeout) clearTimeout(timeout);
      logger.error(
        { group: workspaceRuntime.name, containerName, error: err },
        'Container spawn error',
      );
      resolve({
        status: 'error',
        result: null,
        error: `Container spawn error: ${err.message}`,
      });
    });
  });
}
