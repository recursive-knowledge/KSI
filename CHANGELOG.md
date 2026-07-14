# Changelog

All notable changes to this project should be recorded here, following the
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

ksi is source-only research code (see the "Package distribution" note in
[README.md](./README.md)) and is not published to PyPI, so there is no
released-version history yet — entries accumulate under `[Unreleased]` until
a decision is made to cut and publish releases.

## [Unreleased]

### Changed

- **Renamed the project from KCSI to KSI** (Knowledge-centric Self-Improvement).
  Breaking, no compat shims: the Python package is now `ksi` (was `kcsi`), the
  CLI entry points are `ksi` / `ksi-classify` / `ksi-doctor`, all `KCSI_*`
  environment variables are now `KSI_*`, and the GitHub repo moved to
  `recursive-knowledge/KSI` (old git/web URLs redirect; the docs site moved to
  https://recursive-knowledge.github.io/KSI/ without a redirect). The PyPI-style
  distribution name `knowledge-centric-self-improvement` is unchanged.

### Added

- **Initial public release of KCSI.** Knowledge-centric self-improvement agent
  framework: registry-backed task sources and evaluators, a container-isolated
  runtime, a persistent knowledge substrate (retrieval, discussion,
  distillation, seeding), and bundled benchmark integrations (ARC-AGI 1/2,
  SWE-bench Pro, Polyglot, Terminal-Bench 2).
