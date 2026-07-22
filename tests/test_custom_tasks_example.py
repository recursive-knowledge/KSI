from pathlib import Path

from ksi.tasks.custom import load_custom_tasks

EXAMPLE = Path(__file__).parent.parent / "examples" / "custom_tasks" / "tasks.jsonl"


def test_example_tasks_load_and_are_well_formed():
    tasks = load_custom_tasks(EXAMPLE)
    assert len(tasks) == 4
    for t in tasks:
        assert t.metadata["eval_command"].startswith("python3 ")


# Reference solutions for each demo task. These deliberately-hard tasks are not
# expected to be one-shot for a small model; the references here are the
# known-good solutions used to prove the graders are satisfiable end-to-end.
_REFERENCE_SOLUTIONS = {
    # '/' truncates toward zero (int(a/b)), not Python floor division.
    "calc-eval": r"""
import re
def calc(s):
    toks = re.findall(r"\d+|[-+*/()]", s.replace(" ", ""))
    pos = 0
    def peek():
        return toks[pos] if pos < len(toks) else None
    def eat():
        nonlocal pos
        t = toks[pos]; pos += 1; return t
    def atom():
        t = peek()
        if t == "(":
            eat(); v = expr(); eat(); return v
        if t == "-":
            eat(); return -atom()
        if t == "+":
            eat(); return atom()
        return int(eat())
    def term():
        v = atom()
        while peek() in ("*", "/"):
            op = eat(); r = atom(); v = v * r if op == "*" else int(v / r)
        return v
    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = eat(); r = term(); v = v + r if op == "+" else v - r
        return v
    return expr()
""",
    # Fenwick / binary indexed tree: O(log n) point-set and range-sum.
    "range-queries": r"""
def process(n, ops):
    tree = [0] * (n + 1)
    cur = [0] * n
    def upd(i, d):
        i += 1
        while i <= n:
            tree[i] += d
            i += i & (-i)
    def pre(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s
    out = []
    for op in ops:
        if op[0] == "set":
            _, i, v = op
            upd(i, v - cur[i]); cur[i] = v
        else:
            _, l, r = op
            out.append(pre(r + 1) - pre(l))
    return out
""",
    # Neumaier / Kahan-Babuska compensated summation.
    "precise-sum": r"""
def precise_sum(nums):
    s = 0.0
    c = 0.0
    for x in nums:
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c
""",
    # Nearest-neighbor tour: a valid permutation (score is continuous, so the
    # grader only requires a valid tour to exit 0).
    "tsp-heuristic": r"""
import math
def solve(points):
    n = len(points)
    if n <= 1:
        return list(range(n))
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: d(points[cur], points[j]))
        tour.append(nxt); unvisited.discard(nxt); cur = nxt
    return tour
""",
}


def test_example_evals_fail_on_starter_and_pass_on_reference(tmp_path):
    # The demo must be gradeable end-to-end without an agent: seed each
    # task's files, confirm the eval FAILS pre-solution, then write a
    # reference solution and confirm it PASSES.
    import subprocess

    for task in load_custom_tasks(EXAMPLE):
        seed = Path(task.metadata["repo_path"])
        cmd = task.metadata["eval_command"]
        pre = subprocess.run(cmd, shell=True, cwd=seed, capture_output=True, timeout=120)
        assert pre.returncode != 0, f"{task.id}: eval passed with no solution"
        (seed / "solution.py").write_text(_REFERENCE_SOLUTIONS[task.id], encoding="utf-8")
        post = subprocess.run(cmd, shell=True, cwd=seed, capture_output=True, timeout=120)
        assert post.returncode == 0, f"{task.id}: reference solution failed: {post.stdout} {post.stderr}"
