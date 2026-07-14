"""Drive a kcsi run from Python instead of the CLI.

This is the programmatic equivalent of `scripts/quickstart.sh`: it runs the
bundled synthetic ARC demo tasks through `kcsi.run(...)`. It needs the same
prerequisites as the CLI (Docker running, the `kcsi-agent:bench` image built,
runtime_runner Node deps installed, and a provider API key in the environment).

Run:
    uv run python examples/programmatic/run_arc_demo.py

The point is the wiring, not the result: you build the config + runtime +
evaluator + LLM yourself and hand them to `kcsi.run`, with no argparse and no
CLI in the loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import kcsi
from kcsi.benchmarks import ArcSessionEvaluator
from kcsi.runtime import KcsiContainerExecutor
from kcsi.runtime.llm import build_llm_caller
from kcsi.tasks.loaders import load_tasks_for_source

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_TASKS = REPO_ROOT / "examples" / "quickstart" / "arc_demo" / "demo_recolor.json"


def main() -> None:
    # 1. Load tasks (any registered task source works the same way).
    tasks = load_tasks_for_source(task_source="arc", tasks_path=DEMO_TASKS)

    # 2. Configure the run (programmatic equivalent of the CLI flags).
    config = kcsi.GenerationConfig(
        num_generations=1,
        num_agents=1,
        experiment_name="programmatic_arc_demo",
        knowledge_db_path=str(REPO_ROOT / "runtime_state" / "programmatic_arc_demo_knowledge.sqlite"),
    )

    # 3. Build the pieces yourself — these are the building blocks the CLI uses.
    provider = os.environ.get("MODEL_PROVIDER", "anthropic")
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    llm = build_llm_caller(provider=provider, model=model)
    evaluator = ArcSessionEvaluator()
    runtime = KcsiContainerExecutor(
        # Default host runner command; mirror `--container-command` if you override it.
        command=["npx", "--yes", "--prefix", "runtime_runner", "tsx", "runtime_runner/src/main.ts"],
        working_dir=str(REPO_ROOT),
        knowledge_db_path=config.knowledge_db_path,
        env={k: v for k, v in os.environ.items() if k.startswith(("MODEL", "ANTHROPIC", "OPENAI", "REASONING"))},
    )

    # 4. Run. Returns one TaskTrace per attempt.
    traces = kcsi.run(config, tasks, runtime=runtime, evaluator=evaluator, llm=llm)
    print(f"completed {len(traces)} trace(s)")
    for trace in traces:
        print(f"  task={trace.task_id} native_score={trace.native_score} eval={trace.eval_result}")


if __name__ == "__main__":
    main()
