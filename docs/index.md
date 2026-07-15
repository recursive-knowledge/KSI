---
hide:
  - navigation
  - toc
---

<div class="ksi-hero" markdown>

# Knowledge-centric Self-Improvement

<p class="ksi-tagline">
A generational orchestration framework: a population of disposable agents
attempts **your own tasks**, discusses what worked, distills transferable
knowledge, and seeds the next generation with it.
</p>

<div class="ksi-cta" markdown>
[Get started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[Blog](https://recursive-knowledge.github.io/knowledge-centric-self-improvement/){ .md-button }
[GitHub](https://github.com/recursive-knowledge/KSI){ .md-button }
</div>

</div>

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg } &nbsp; **Getting started**

    ---

    Fresh clone to a solved demo task in one command — no dataset download.

    [:octicons-arrow-right-24: Run the demo](getting-started.md)

-   :material-file-document-edit:{ .lg } &nbsp; **Your own tasks**

    ---

    Point KSI at any JSON/JSONL file of tasks — the record schema and the
    `command` evaluator's scoring contract.

    [:octicons-arrow-right-24: Bring your own tasks](your_own_tasks.md)

-   :material-frequently-asked-questions:{ .lg } &nbsp; **FAQ**

    ---

    What is this, when to use it, what it costs, and where results go.

    [:octicons-arrow-right-24: Browse the FAQ](faq.md)

-   :material-language-python:{ .lg } &nbsp; **Programmatic API**

    ---

    Drive a run from Python with `ksi.run(...)` — plus
    [larger runs](experiments.md) and the
    [API reference](reference/api.md).

    [:octicons-arrow-right-24: API guide](programmatic_api.md)

-   :material-sitemap:{ .lg } &nbsp; **Architecture**

    ---

    The runtime execution chain and which component owns each database.

    [:octicons-arrow-right-24: Read the overview](architecture.md)

-   :material-puzzle:{ .lg } &nbsp; **Extending KSI**

    ---

    The four `register_*` extension seams, adding benchmarks and
    evaluators, and the [improvement strategies](improvement_strategies.md).

    [:octicons-arrow-right-24: Extend KSI](extending.md)

-   :material-book-open-variant:{ .lg } &nbsp; **Reference**

    ---

    The [glossary](glossary.md), [artifacts & cleanup](artifacts.md), and
    runtime startup performance notes.

    [:octicons-arrow-right-24: Browse the reference](glossary.md)

-   :material-chart-bar:{ .lg } &nbsp; **Benchmarks**

    ---

    Run the reference benchmarks yourself — data prep, task maps, and
    run presets.

    [:octicons-arrow-right-24: See benchmarks](benchmarks.md)

</div>

!!! note
    The files under `docs/` remain the canonical source; this site renders them
    plus an auto-generated [Python API reference](reference/api.md).
