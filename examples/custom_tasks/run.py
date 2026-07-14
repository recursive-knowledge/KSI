"""Drive a ksi run on the bundled custom-tasks demo from Python.

This is the programmatic equivalent of `run.sh`: it runs the three
self-contained custom tasks in `tasks.jsonl` through `ksi.run(...)`. It
needs the same prerequisites as the CLI (Docker running, the
`ksi-agent:bench` image built, runtime_runner Node deps installed, and a
provider API key in the environment).

Run:
    uv run python examples/custom_tasks/run.py

The point is the wiring, not the result: you build the config + runtime +
evaluator + LLM yourself and hand them to `ksi.run`, with no argparse and no
CLI in the loop.
"""

from __future__ import annotations

from pathlib import Path

import ksi
from ksi.eval.command import CommandEvaluator
from ksi.providers import load_provider_profile
from ksi.runtime import KsiContainerExecutor
from ksi.runtime.llm import build_llm_caller
from ksi.tasks.loaders import load_tasks_for_source

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_TASKS = REPO_ROOT / "examples" / "custom_tasks" / "tasks.jsonl"


def main() -> None:
    # 1. Load tasks (any registered task source works the same way).
    tasks = load_tasks_for_source(task_source="custom", tasks_path=DEMO_TASKS)

    # 2. Configure the run (programmatic equivalent of the CLI flags).
    config = ksi.GenerationConfig(
        num_generations=1,
        num_agents=1,
        experiment_name="programmatic_custom_demo",
        knowledge_db_path=str(REPO_ROOT / "runtime_state" / "programmatic_custom_demo_knowledge.sqlite"),
    )

    # 3. Build the pieces yourself — these are the building blocks the CLI uses.
    provider_env = load_provider_profile("configs/ksi/.env.haiku")
    llm = build_llm_caller(provider=provider_env["MODEL_PROVIDER"], model=provider_env["MODEL"])
    evaluator = CommandEvaluator()
    runtime = KsiContainerExecutor(
        # Default host runner command; mirror `--container-command` if you override it.
        command=["npx", "--yes", "--prefix", "runtime_runner", "tsx", "runtime_runner/src/main.ts"],
        working_dir=str(REPO_ROOT),
        knowledge_db_path=config.knowledge_db_path,
        env=provider_env,
    )

    # 4. Run. Returns one TaskTrace per attempt.
    traces = ksi.run(config, tasks, runtime=runtime, evaluator=evaluator, llm=llm)
    print(f"completed {len(traces)} trace(s)")
    for trace in traces:
        print(f"  task={trace.task_id} native_score={trace.native_score} eval={trace.eval_result}")


if __name__ == "__main__":
    main()
