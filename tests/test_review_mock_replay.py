import sys
import tarfile
from dataclasses import replace
from io import BytesIO

import pytest
from test_executor import FixtureRunner

from github_harness_agent.executor import Limits, PlainExecutor, TaskSnapshot, _Budget
from github_harness_agent.review_mock_probe import decode_traces
from github_harness_agent.review_mock_replay import mock_call_diagnostic
from github_harness_agent.review_solver import patch_for


_TEST = """\
from unittest.mock import Mock
from service import invoke

class Opaque:
    def __repr__(self):
        raise RuntimeError("do not format opaque objects")

def test_invoke():
    mock_sink = Mock()
    excluded = Mock()
    excluded(**{"unsafe keyword name": 8})
    invoke(mock_sink, Opaque())
    assert len(mock_sink.mock_calls) == 2
    assert mock_sink.mock_calls[0].args[0] == (1, True)

def test_unselected():
    raise AssertionError("only the target test may run")
"""
_BASE = """\
def invoke(sink, opaque):
    sink.send((1, True), "secret argument", opaque, active=False, bogus=float("nan"))
    sink.accept(3)
"""
_COMMAND = (sys.executable, "-m", "pytest", "-q", "tests/test_service.py")
_NODE = "tests/test_service.py::test_invoke"


def _archive(source):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as snapshot:
        for path, text in {
            "service.py": source,
            "tests/test_service.py": _TEST,
        }.items():
            data = text.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "case",
    ["numeric difference", "unchanged calls", "baseline fails", "candidate passes"],
)
def test_mock_call_probe_checks_exact_node_and_only_reports_safe_differences(
    tmp_path, case
):
    before = (
        _BASE if case != "baseline fails" else _BASE.replace("(1, True)", "(9, True)")
    )
    if case == "numeric difference":
        after = before.replace("(1, True)", "(2, True)")
    elif case == "candidate passes":
        after = before.replace("accept(3)", "accept(4)")
    else:
        after = before + '    print("secret terminal line")\n    assert False\n'
    runner = FixtureRunner(tmp_path)
    executor = PlainExecutor(runner, None, Limits(), verification_runner=runner)
    task = TaskSnapshot("org/fixture", "1" * 40, "Fix invoke.", (_COMMAND,))
    budget = _Budget(executor.limits, lambda: False, task)
    patch = patch_for({"service.py": before}, {"service.py": after})

    detail = mock_call_diagnostic(
        executor, task, _archive(before), budget, _COMMAND, _NODE, patch
    )

    if case == "numeric difference":
        assert "quoted advisory data" in detail
        assert "mock_sink" in detail and "send" in detail
        assert "[1,true]" in detail and "[2,true]" in detail
        assert '"active":false' in detail
        assert "secret" not in detail and "Opaque" not in detail and "NaN" not in detail
        assert "unsafe keyword name" not in detail and "excluded" not in detail
    else:
        assert detail == ""
    assert len(runner.workspaces) == (1 if case == "baseline fails" else 2)


def test_mock_call_probe_skips_authorized_edit_to_selected_test(tmp_path):
    runner = FixtureRunner(tmp_path)
    executor = PlainExecutor(runner, None, Limits(), verification_runner=runner)
    task = replace(
        TaskSnapshot("org/fixture", "1" * 40, "Fix invoke.", (_COMMAND,)),
        test_update="tests/test_service.py::test_invoke",
    )
    budget = _Budget(executor.limits, lambda: False, task)
    patch = patch_for(
        {"service.py": _BASE}, {"service.py": _BASE.replace("(1, True)", "(2, True)")}
    )

    assert (
        mock_call_diagnostic(
            executor, task, _archive(_BASE), budget, _COMMAND, _NODE, patch
        )
        == ""
    )
    assert runner.workspaces == []


@pytest.mark.parametrize(
    "output",
    [
        b'{"mock":[["",["arbitrary text"],{}]]}',
        b'{"mock":[["",[],{"unsafe keyword":1}]]}',
        b'{"mock":[["",[],{"value":1e500}]]}',
        b'{"mock":[["",[[[[1]]]],{}]]}',
    ],
)
def test_untrusted_mock_report_must_match_safe_schema(output):
    assert decode_traces(output) is None
