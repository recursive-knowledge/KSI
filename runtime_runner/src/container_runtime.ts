/**
 * Container runtime abstraction for the KSI shared runtime.
 * All container-runtime-specific logic lives here so swapping runtimes means
 * changing one file.
 */
/** The container runtime binary name. */
export const CONTAINER_RUNTIME_BIN = 'docker';

/** Returns CLI args for a readonly bind mount. */
export function readonlyMountArgs(
  hostPath: string,
  containerPath: string,
): string[] {
  return ['-v', `${hostPath}:${containerPath}:ro`];
}

/**
 * Returns the argv for stopping a container by name, for use with `execFile`
 * (NOT a shell string). Returning `string[]` keeps the container name out of a
 * shell — the name is sanitized to `[a-zA-Z0-9-]+` upstream so this is latent,
 * not exploitable today, but argv form removes the injection surface entirely.
 */
export function stopContainer(name: string): string[] {
  return ['stop', name];
}

