/**
 * Copy-sync guard helpers for the inline JS copies of TypeScript helpers
 * under tests/js/ (issue #734).
 *
 * The repo's JS test harness runs Node directly (no tsc step), so several
 * tests keep inline JS copies of agent-runner TypeScript helpers. Those
 * copies have drifted before (silent_failure_recovery.test.mjs was missing
 * the `iterator_drain_pending_tools` trigger branch and the emitPhrase
 * marker sourcing). This module makes the "must stay in sync" comments
 * enforceable:
 *
 *   - `extractFunction(source, name)`: brace-extracts `function NAME(...)`
 *     from a source file (params + body), skipping strings / template
 *     literals / comments while balancing.
 *   - `normalizeFunction(...)`: strips TypeScript type annotations
 *     conservatively (variable-declaration annotations, single-param arrow
 *     annotations, arrow/function return annotations, `as X` casts, `!.`
 *     non-null assertions), removes comments, canonicalizes quote style for
 *     simple strings, and collapses whitespace — yielding a canonical form
 *     in which a faithful JS copy is byte-identical to its TS source.
 *   - `compareFunctions(...)`: extracts + normalizes the named function from
 *     a TS source and a JS source and reports the first divergence.
 *
 * KNOWN, DOCUMENTED LIMITS (see tests/js/copy_sync_guard.test.mjs for the
 * compensating anchor assertions):
 *   - Parameter LISTS are compared by parameter NAME only. Parameter type
 *     annotations and parameter DEFAULT VALUES are not compared (stripping a
 *     function-typed annotation like `readFileImpl: (p: string) => string =
 *     (p) => ...` is exactly the brittle case this avoids). Critical
 *     defaults are pinned by anchor assertions instead.
 *   - Function RETURN type annotations are skipped during extraction, never
 *     compared.
 *   - The type stripper handles only the constructs present in the guarded
 *     helpers (no regex literals, no `satisfies`, no decorators). New TS
 *     syntax in a guarded helper may need a stripper extension — the guard
 *     then fails loudly rather than passing silently.
 */

/**
 * Split source text into segments so later passes can transform code
 * without touching string/template-literal contents.
 * Returns [{ kind: 'code'|'string'|'template'|'comment', text }].
 * String segments include their quotes; template segments include the
 * backticks and the `${` / `}` delimiters; `${...}` expression interiors
 * are code segments. Does NOT handle regex literals (none exist in the
 * guarded helpers).
 */
export function segments(source) {
  const segs = [];
  let buf = "";
  let kind = "code";
  // Stack entries: {type:'tpl'} for a template literal, {type:'expr',depth}
  // for a ${...} expression inside a template.
  const stack = [];
  const flush = (nextKind) => {
    if (buf) segs.push({ kind, text: buf });
    buf = "";
    kind = nextKind;
  };
  let i = 0;
  const n = source.length;
  while (i < n) {
    const c = source[i];
    if (kind === "code") {
      const c2 = source.slice(i, i + 2);
      if (c2 === "//") {
        flush("comment");
        while (i < n && source[i] !== "\n") {
          buf += source[i];
          i += 1;
        }
        flush("code");
        continue;
      }
      if (c2 === "/*") {
        flush("comment");
        const end = source.indexOf("*/", i + 2);
        const stop = end === -1 ? n : end + 2;
        buf = source.slice(i, stop);
        i = stop;
        flush("code");
        continue;
      }
      if (c === "'" || c === '"') {
        flush("string");
        buf += c;
        i += 1;
        while (i < n) {
          const s = source[i];
          buf += s;
          i += 1;
          if (s === "\\") {
            buf += source[i] ?? "";
            i += 1;
            continue;
          }
          if (s === c) break;
        }
        flush("code");
        continue;
      }
      if (c === "`") {
        flush("template");
        buf += c;
        i += 1;
        stack.push({ type: "tpl" });
        continue;
      }
      const top = stack[stack.length - 1];
      if (top && top.type === "expr") {
        if (c === "{") {
          top.depth += 1;
        } else if (c === "}") {
          if (top.depth > 0) {
            top.depth -= 1;
          } else {
            stack.pop();
            flush("template");
            buf += "}";
            i += 1;
            continue;
          }
        }
      }
      buf += c;
      i += 1;
      continue;
    }
    if (kind === "template") {
      if (c === "\\") {
        buf += c + (source[i + 1] ?? "");
        i += 2;
        continue;
      }
      if (source.slice(i, i + 2) === "${") {
        buf += "${";
        i += 2;
        stack.push({ type: "expr", depth: 0 });
        flush("code");
        continue;
      }
      if (c === "`") {
        buf += c;
        i += 1;
        stack.pop();
        flush("code");
        continue;
      }
      buf += c;
      i += 1;
      continue;
    }
    // 'string'/'comment' segments are consumed inline above; never reached.
    throw new Error(`segments(): unexpected scanner state '${kind}'`);
  }
  flush("code");
  return segs;
}

/** Offsets of code-only regions, for scanner-aware brace balancing. */
function codeRanges(source) {
  const ranges = [];
  let offset = 0;
  for (const seg of segments(source)) {
    if (seg.kind === "code") {
      ranges.push([offset, offset + seg.text.length]);
    }
    offset += seg.text.length;
  }
  return ranges;
}

function inCode(ranges, idx) {
  return ranges.some(([a, b]) => idx >= a && idx < b);
}

/** Index of the matching close delimiter, counting only code regions. */
function findMatching(source, ranges, openIdx, openCh, closeCh) {
  let depth = 0;
  for (let i = openIdx; i < source.length; i += 1) {
    if (!inCode(ranges, i)) continue;
    if (source[i] === openCh) depth += 1;
    else if (source[i] === closeCh) {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  throw new Error(`findMatching(): unbalanced '${openCh}' at ${openIdx}`);
}

/** Skip whitespace and comments starting at idx; returns the next code idx. */
function skipWsAndComments(source, idx) {
  let i = idx;
  for (;;) {
    while (i < source.length && /\s/.test(source[i])) i += 1;
    if (source.slice(i, i + 2) === "//") {
      while (i < source.length && source[i] !== "\n") i += 1;
      continue;
    }
    if (source.slice(i, i + 2) === "/*") {
      const end = source.indexOf("*/", i + 2);
      i = end === -1 ? source.length : end + 2;
      continue;
    }
    return i;
  }
}

/**
 * Brace-extract `function NAME(...)` from `source`. Skips an optional
 * `: ReturnType` annotation (including inline object types) between the
 * parameter list and the body. Returns { params, body, start, end } where
 * `params` is the raw text between the parens and `body` the raw text
 * between the outermost body braces.
 */
export function extractFunction(source, name) {
  const m = source.match(
    new RegExp(`(?:^|[^A-Za-z0-9_$.])function\\s+${name}\\s*\\(`),
  );
  if (!m) {
    throw new Error(`extractFunction(): 'function ${name}(' not found`);
  }
  const ranges = codeRanges(source);
  const parenOpen = m.index + m[0].length - 1;
  const parenClose = findMatching(source, ranges, parenOpen, "(", ")");
  let i = skipWsAndComments(source, parenClose + 1);
  if (source[i] === ":") {
    // Return-type annotation. Either an inline object type `{...}` or a
    // brace-free type expression (identifier, union, generic w/o braces).
    i = skipWsAndComments(source, i + 1);
    if (source[i] === "{") {
      i = findMatching(source, ranges, i, "{", "}") + 1;
      i = skipWsAndComments(source, i);
    } else {
      while (i < source.length && source[i] !== "{") i += 1;
    }
  }
  if (source[i] !== "{") {
    throw new Error(`extractFunction(): body '{' not found for ${name}`);
  }
  const bodyClose = findMatching(source, ranges, i, "{", "}");
  return {
    params: source.slice(parenOpen + 1, parenClose),
    body: source.slice(i + 1, bodyClose),
    start: m.index,
    end: bodyClose + 1,
  };
}

/** Parameter names (annotations/defaults ignored; see module docs). */
export function paramNames(paramsText) {
  const cleaned = segments(paramsText)
    .map((s) => (s.kind === "comment" ? " " : s.text))
    .join("");
  const parts = [];
  let depth = 0;
  let cur = "";
  let prev = "";
  for (const ch of cleaned) {
    if ("({[".includes(ch)) depth += 1;
    else if (")}]".includes(ch)) depth -= 1;
    // Generic type args, e.g. `sdkEnv: Record<string, string | undefined>` —
    // a top-level comma inside `<...>` must not split the parameter list.
    // Only count `<` preceded by an identifier char (a generic instantiation
    // like `Record<`), and never count the `>` of an arrow (`=>`), so this
    // doesn't misfire on arrow-typed defaults elsewhere in these signatures.
    else if (ch === "<" && /[A-Za-z0-9_$]/.test(prev)) depth += 1;
    else if (ch === ">" && prev !== "=" && depth > 0) depth -= 1;
    if (ch === "," && depth === 0) {
      parts.push(cur);
      cur = "";
      prev = ch;
      continue;
    }
    cur += ch;
    prev = ch;
  }
  parts.push(cur);
  return parts
    .map((p) => (p.match(/^\s*(?:\.\.\.)?([A-Za-z_$][\w$]*)/) || [])[1])
    .filter(Boolean);
}

/**
 * Conservative TS-annotation stripping for a CODE segment (never applied to
 * string/template text). Handles exactly the constructs in the guarded
 * helpers; see module docs.
 */
function stripTypesFromCode(code) {
  let out = code;
  // `as { ... }` object-literal casts, e.g. `(e as { cause?: unknown })`.
  out = out.replace(/\s+as\s+\{[^{}]*\}/g, "");
  // `as Name`, `as ns.Name`, `as Name<...>` (one nesting level), `as T[]`.
  out = out.replace(
    /\s+as\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:<[^<>]*(?:<[^<>]*>)?[^<>]*>)?(?:\[\])*/g,
    "",
  );
  // Non-null assertions before property access: `x!.y` -> `x.y`.
  out = out.replace(/!(?=\.)/g, "");
  // Variable-declaration annotations: `let x: T = ...` / `let x: T;`.
  out = out.replace(
    /\b(const|let|var)\s+([A-Za-z_$][\w$]*)\s*:\s*[^=;]+?(?=[=;])/g,
    "$1 $2 ",
  );
  // Regular function parameter annotations/defaults are intentionally ignored
  // by the sync contract; compare parameter names only.
  out = out.replace(
    /\bfunction\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)/g,
    (match, name, params) => {
      const names = paramNames(params);
      if (names.length === 0 && params.trim()) return match;
      return `function ${name}(${names.join(",")})`;
    },
  );
  // Single-parameter arrow annotations: `(v: unknown)` -> `(v)`.
  out = out.replace(/\(\s*([A-Za-z_$][\w$]*)\s*:\s*[A-Za-z_$][\w$.]*(?:\[\])?\s*\)/g, "($1)");
  // Arrow/function return annotations: `): T => ` / `): T {` (generics/unions ok).
  out = out.replace(
    /\)\s*:\s*[^{}=;]+?\s*(=>|\{)/g,
    ") $1",
  );
  // Redundant parens left by cast removal: `(e).cause` -> `e.cause`.
  // Symmetric on both sides (also turns `(v) =>` into `v =>` everywhere).
  out = out.replace(/(?<![\w$.)\]])\(([A-Za-z_$][\w$.]*)\)/g, "$1");
  return out;
}

/** Canonical quote style: simple strings become single-quoted. */
function canonString(text) {
  const inner = text.slice(1, -1);
  if (/['"\\]/.test(inner)) return text;
  return `'${inner}'`;
}

/** Per-code-segment whitespace canonicalization. */
function collapseCode(code) {
  return code
    .replace(/\s+/g, " ")
    .replace(/ ?([{}()[\];,.:<>=+\-|&!?*]) ?/g, "$1");
}

/**
 * Normalize a function body (or any code snippet): strip comments + TS
 * annotations, canonicalize quotes and whitespace. Template-literal text is
 * preserved byte-for-byte (marker phrases must match exactly).
 */
export function normalizeBody(bodyText) {
  // Pre-pass (before segmentation): indexed-access type casts such as
  // `status as ContainerOutput['status']` span a string boundary — the
  // `'status'` quotes are isolated into a string segment, so a per-code-segment
  // stripper can't see the whole cast. Strip them here on the raw text. Safe
  // because this exact shape only appears as a TS cast in the guarded helpers,
  // never inside a preserved marker literal.
  bodyText = bodyText.replace(
    /\s+as\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\[\s*['"][^'"]*['"]\s*\]/g,
    "",
  );
  // Map comments to a space and merge adjacent code pieces BEFORE the
  // whitespace collapse, so a comment present on only one side cannot leave
  // an asymmetric double-space at a segment boundary.
  const pieces = [];
  for (const seg of segments(bodyText)) {
    if (seg.kind === "comment") {
      pieces.push({ lit: false, text: " " });
    } else if (seg.kind === "string") {
      pieces.push({ lit: true, text: canonString(seg.text) });
    } else if (seg.kind === "template") {
      pieces.push({ lit: true, text: seg.text });
    } else {
      pieces.push({ lit: false, text: stripTypesFromCode(seg.text) });
    }
  }
  let out = "";
  let codeBuf = "";
  for (const piece of pieces) {
    if (piece.lit) {
      out += collapseCode(codeBuf);
      codeBuf = "";
      out += piece.text;
    } else {
      codeBuf += piece.text;
    }
  }
  out += collapseCode(codeBuf);
  return out.trim();
}

/** First divergence between two normalized strings, with context. */
function firstDiff(a, b, context = 90) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i += 1;
  if (i === n && a.length === b.length) return null;
  const lo = Math.max(0, i - context);
  return (
    `first divergence at normalized offset ${i}:\n` +
    `  ts: ...${a.slice(lo, i + context)}...\n` +
    `  js: ...${b.slice(lo, i + context)}...`
  );
}

/**
 * Compare `function name(...)` between a TS source and a JS source.
 * Returns { ok: true } or { ok: false, message } with a readable report.
 * `jsName` defaults to `name` (copies keep the TS name).
 */
export function compareFunctions(tsSource, jsSource, name, { jsName = name } = {}) {
  let ts;
  let js;
  try {
    ts = extractFunction(tsSource, name);
  } catch (err) {
    return { ok: false, message: `TS side: ${err.message}` };
  }
  try {
    js = extractFunction(jsSource, jsName);
  } catch (err) {
    return { ok: false, message: `JS side: ${err.message}` };
  }
  const tsParams = paramNames(ts.params);
  const jsParams = paramNames(js.params);
  if (JSON.stringify(tsParams) !== JSON.stringify(jsParams)) {
    return {
      ok: false,
      message:
        `${name}: parameter names differ — ` +
        `ts=${JSON.stringify(tsParams)} js=${JSON.stringify(jsParams)}`,
    };
  }
  const tsNorm = normalizeBody(ts.body);
  const jsNorm = normalizeBody(js.body);
  if (tsNorm !== jsNorm) {
    return {
      ok: false,
      message:
        `${name}: body drifted from the TS source — update the inline copy ` +
        `to mirror the TS (types stripped). ${firstDiff(tsNorm, jsNorm)}`,
    };
  }
  return { ok: true };
}
