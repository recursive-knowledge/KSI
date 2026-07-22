from pathlib import Path

from ksi.tasks.custom import load_custom_tasks

EXAMPLE = Path(__file__).parent.parent / "examples" / "custom_tasks" / "tasks.jsonl"


def test_example_tasks_load_and_are_well_formed():
    tasks = load_custom_tasks(EXAMPLE)
    assert len(tasks) == 3
    for t in tasks:
        assert t.metadata["eval_command"].startswith("python3 ")


def test_example_evals_fail_on_starter_and_pass_on_reference(tmp_path):
    # The demo must be gradeable end-to-end without an agent: seed each
    # task's files, confirm the eval FAILS pre-solution, then write a
    # reference solution and confirm it PASSES.
    import subprocess

    solutions = {
        "fizzbuzz": "def fizzbuzz(n):\n    return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1, n+1)]\n",
        "reverse-words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
        "anagram-groups": "def group_anagrams(words):\n    groups = {}\n    for w in words:\n        groups.setdefault(''.join(sorted(w)), []).append(w)\n    return sorted([sorted(g) for g in groups.values()], key=lambda g: g[0])\n",
    }
    for task in load_custom_tasks(EXAMPLE):
        seed = Path(task.metadata["repo_path"])
        cmd = task.metadata["eval_command"]
        pre = subprocess.run(cmd, shell=True, cwd=seed, capture_output=True, timeout=60)
        assert pre.returncode != 0, f"{task.id}: eval passed with no solution"
        (seed / "solution.py").write_text(solutions[task.id], encoding="utf-8")
        post = subprocess.run(cmd, shell=True, cwd=seed, capture_output=True, timeout=60)
        assert post.returncode == 0, f"{task.id}: reference solution failed: {post.stdout} {post.stderr}"
