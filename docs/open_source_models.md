# Open-source and self-hosted models

Can KSI run against Llama, Qwen, DeepSeek, or a model you host yourself on
vLLM/Ollama? **Not out of the box today**, but the gap is wiring rather than
architecture. This page sketches how it would work, what it touches, and the
problems to expect. Contributions welcome.

## Where things stand

`MODEL_PROVIDER` accepts exactly two values — `anthropic` and `openai`. There is
no `vllm`, `ollama`, or `openrouter` provider, and an unrecognized value is
rejected up front rather than silently falling back.

## How you would wire it

Don't write a new provider adapter. Both SDKs KSI uses are built to be
repointed at a different endpoint, so the shortest path is to put a **proxy that
speaks the Anthropic Messages format** in front of your model —
[LiteLLM](https://docs.litellm.ai/docs/anthropic_unified/) translates that
format to most backends — and let the existing Claude Agent SDK path talk to it.
The SDK honors `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN`, so the whole
agent loop (multi-turn, native tools, the MCP memory server, hooks) keeps
working unchanged. `MODEL_PROVIDER` stays `anthropic`; `MODEL` becomes whatever
name your proxy routes.

Pointing at an OpenAI-compatible endpoint instead is also possible, but it is
more work and needs care about which API shape your server implements — many
implement Chat Completions only, and a server that does expose a Responses
endpoint may still not handle the multi-turn history an agent loop replays
through it.

This split is why it half-works today:

- **Host-side phases** — forum, distillation, reflection, task-claiming — already
  honor the base-URL environment variables, so they can be redirected right now.
- **The containerized agent** cannot. Task execution runs in a Docker container
  whose environment is built from a fixed set of keys, and the base-URL variables
  aren't among them. Getting them forwarded is the change a contributor would
  need to make.

!!! warning "Don't half-configure this"
    Setting a base URL today does not error — it **splits your traffic**:
    knowledge phases hit your local server while every task container still calls
    the hosted API, billing your key or failing on the egress allowlist. Treat
    the current state as useful for experimenting with the knowledge phases only.

## What it affects

- **Egress isolation.** Agent containers reach the network only through an
  allowlisting proxy (see
  [Architecture § Egress isolation](./architecture.md#10-egress-isolation)), so a
  self-hosted endpoint has to be allowlisted. A model server on the Docker host
  is its own problem: `localhost` inside the container is the container, and the
  internal network has no route back to the host.
- **Prompt caching.** KSI places cache breakpoints on stable prompt prefixes.
  Whether those survive a proxy translation varies, so expect cache savings — and
  the cache columns in token accounting — to largely disappear.
- **Cost reporting.** Unrecognized model names price at `$0.00`. Reasonable for a
  self-hosted model, but it makes cost comparisons against a hosted baseline
  misleading.
- **Structured output.** Forum, distillation, and task-claiming ask for
  schema-constrained JSON. There is already a fallback for callers that can't
  provide it, so this degrades rather than breaks — but whether the looser output
  holds up in quality is an open question.
- **Direct adapters.** A couple of paths (forum, ARC) bypass the SDK and call the
  Anthropic API directly against a hardcoded URL. They would need the same
  treatment, or to be configured back onto the SDK path.

## Problems to expect

The plumbing is the easy part. The real risk is **tool-calling robustness**: the
agent drives a long multi-turn loop with native tools and an MCP server attached,
and smaller open models tend to fail there — dropping tool calls, malforming
arguments, or looping — long before any of the above matters. Reasoning-effort
handling is also currently keyed to specific hosted model families, so
reasoning-capable open models would need that revisited.

Worth validating cheaply before investing: redirect the host-side phases (which
works today) and see whether one forum round and one distillation produce usable
output. If they don't, the container work won't save it.

## If you just want a cheap model

If the goal is lower cost rather than open weights specifically, the supported
route is a small hosted model — `.env.haiku.template` or `.env.openai.template`.
See the [FAQ](./faq.md#what-models-and-providers-can-i-use-do-i-need-an-api-key)
for the full list of bundled profile templates.
