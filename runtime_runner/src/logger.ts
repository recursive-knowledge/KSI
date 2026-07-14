import pino from 'pino';

const level = process.env.LOG_LEVEL || process.env.KCSI_LOG_LEVEL || 'info';
const enablePretty = process.env.KCSI_PRETTY_LOGS === '1';
const loggerConfig = enablePretty
  ? { level, transport: { target: 'pino-pretty', options: { colorize: true, destination: 2 } } }
  : { level };

// IMPORTANT: pino defaults to stdout (fd 1). The KCSI host process uses
// stdout EXCLUSIVELY as a single-shot channel for the final envelope JSON
// that ``src/kcsi/runtime/normalize.py::parse_runner_stdout`` parses. Writing
// pino log lines to the same fd interleaves them with the envelope on
// pipes (Node's ``process.stdout.write`` is not atomic for writes larger
// than PIPE_BUF / 4096 bytes on Linux), which corrupts the envelope mid-
// string and forces the host to fall back to the raw-text path. Observed
// in the arc2 v3 memory-enabled run where 36/60 attempts (60%) had a pino
// log injected into the middle of the envelope. We redirect pino to
// stderr (fd 2) so the only stdout writer is the single envelope emission
// at the end of ``runtime_runner/src/main.ts``.
export const logger = pino(loggerConfig, pino.destination(2));

// Route uncaught errors through pino so they get timestamps in stderr
process.on('uncaughtException', (err) => {
  logger.fatal({ err }, 'Uncaught exception');
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  logger.error({ err: reason }, 'Unhandled rejection');
});
