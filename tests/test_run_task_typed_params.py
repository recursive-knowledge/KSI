import inspect

from kcsi.protocols import RuntimeExecutor
from kcsi.runtime.container_host import KcsiContainerExecutor
from kcsi.runtime.terminal_bench_2 import TerminalBench2Executor


def _params(fn):
    return inspect.signature(fn).parameters


def test_protocol_declares_cross_task_params():
    p = _params(RuntimeExecutor.run_task)
    assert p["cross_task_shared_container"].default is False
    assert "cross_task_r1_callback" in p
    assert any(x.kind == x.VAR_KEYWORD for x in p.values())  # **kwargs retained


def test_container_host_and_tb2_impls_declare_them():
    for cls in (KcsiContainerExecutor, TerminalBench2Executor):
        p = _params(cls.run_task)
        assert "cross_task_shared_container" in p
        assert "cross_task_r1_callback" in p
        assert any(x.kind == x.VAR_KEYWORD for x in p.values())  # **kwargs retained
