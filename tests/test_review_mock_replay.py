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


def _archive(source, test=_TEST):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as snapshot:
        for path, text in {
            "service.py": source,
            "tests/test_service.py": test,
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


_COUNTER_SOURCE = """
class Collector:
    def __init__(self):
        self.summary = {"processed_count": 0, "private_text": "never print me"}

    def process(self):
        self._tick()
        return self.summary

    def _tick(self):
        self.summary["processed_count"] += 1

def invoke():
    return Collector().process()
"""
_COUNTER_TEST = """
from service import invoke

def test_invoke():
    assert invoke()["processed_count"] == 1
"""


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("nested", True),
        ("empty baseline", True),
        ("unreviewed", False),
        ("ambiguous", False),
        ("ambiguous mappings", False),
        ("unrelated mutation", False),
        ("unchanged transitions", False),
        ("sensitive field", False),
    ],
)
def test_replay_reports_bounded_counter_locations_only(tmp_path, case, expected):
    before, test = _COUNTER_SOURCE, _COUNTER_TEST
    after = before.replace("+= 1", "+= 2")
    paths = ("service.py",)
    if case == "empty baseline":
        before = before.replace('self.summary["processed_count"] += 1', "pass")
        before = before.replace('"processed_count": 0', '"processed_count": 1')
        after = before.replace(
            "        pass", '        self.summary["processed_count"] += 1'
        )
    elif case == "unreviewed":
        paths = ("unreviewed.py",)
    elif case == "ambiguous":
        test += '    assert invoke()["processed_count"] == 1\n'
    elif case == "ambiguous mappings":
        before = before.replace(
            "        self._tick()",
            '        self.shadow = {"processed_count": 0}\n        self._tick()',
        ).replace(
            '        self.summary["processed_count"] += 1',
            '        self.summary["processed_count"] += 1\n        self.shadow["processed_count"] += 1',
        )
        after = before.replace("+= 1", "+= 2")
    elif case == "unrelated mutation":
        after = before.replace("+= 1", "+= 8").replace(
            "        return self.summary", '        return {"processed_count": 2}'
        )
    elif case == "unchanged transitions":
        after = before.replace("return self.summary", 'return {"processed_count": 2}')
    elif case == "sensitive field":
        before, after, test = (
            source.replace("processed_count", "access_token")
            for source in (before, after, test)
        )
    runner = FixtureRunner(tmp_path)
    executor = PlainExecutor(runner, None, Limits(), verification_runner=runner)
    task = TaskSnapshot(
        "org/fixture", "1" * 40, "Exercise public processing.", (_COMMAND,)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    patch = patch_for({"service.py": before}, {"service.py": after})
    detail = mock_call_diagnostic(
        executor,
        task,
        _archive(before, test),
        budget,
        _COMMAND,
        _NODE,
        patch,
        reviewed_paths=paths,
    )
    if not expected:
        assert detail == ""
        return
    assert "mapping counter" in detail
    assert "Collector._tick" in detail and "Collector.process" in detail
    assert "processed_count" not in detail and "never print me" not in detail
    assert "0 -> 2" in detail if case == "nested" else "1 -> 2" in detail
    if case == "empty baseline":
        assert "no changes observed" in detail
    assert len(runner.workspaces) == 2
    assert all(
        (workspace / "tests/test_service.py").read_text() == test
        for workspace in runner.workspaces
    )


@pytest.mark.parametrize(
    ("selected_line", "changed_test_path", "expected"),
    [
        ("unset", False, False),
        ("matching", False, True),
        ("matching", True, False),
        ("wrong", False, False),
    ],
)
def test_counter_replay_pairs_fresh_objects_at_the_selected_occurrence(
    tmp_path, selected_line, changed_test_path, expected
):
    before = """class Collector:
    def __init__(self):
        self.summary = {"processed_count": 0, "other_count": 1}

    def process(self, mode):
        self._tick(mode)
        return self.summary

    def _tick(self, mode):
        self.summary["processed_count"] += 1 if mode != 9 else 7

def invoke(mode):
    return Collector().process(mode)
"""
    after = before.replace(
        "1 if mode != 9 else 7", "2 if mode == 2 else (1 if mode != 9 else 7)"
    )
    if changed_test_path:
        after = after.replace(
            "        self._tick(mode)",
            '        self._tick(mode)\n        if mode == 2:\n            self.summary["other_count"] = 0',
        )
    test = """from service import invoke

def test_invoke():
    for mode in (1, 2):
        result = invoke(mode)
        if result["other_count"]:
            placeholder = 1
        assert result["processed_count"] == 1
    assert invoke(9)["processed_count"] == 7
"""
    line = test.splitlines().index('        assert result["processed_count"] == 1') + 1
    runner = FixtureRunner(tmp_path)
    executor = PlainExecutor(runner, None, Limits(), verification_runner=runner)
    task = TaskSnapshot("org/fixture", "1" * 40, "Exercise processing.", (_COMMAND,))
    budget = _Budget(executor.limits, lambda: False, task)
    detail = mock_call_diagnostic(
        executor,
        task,
        _archive(before, test),
        budget,
        _COMMAND,
        _NODE,
        patch_for({"service.py": before}, {"service.py": after}),
        reviewed_paths=("service.py",),
        failure_line={"matching": line, "wrong": line + 1}.get(selected_line),
    )
    if expected:
        assert f"{_NODE} line {line}" in detail
        assert "evaluation 2" in detail
        assert "0 -> 1" in detail and "0 -> 2" in detail
        assert "0 -> 7" not in detail and "processed_count" not in detail
        assert "Collector._tick" in detail
    else:
        assert detail == ""
