"""Run exactly one tracked pytest node and expose bounded assertion metadata."""

import ast
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

REPORT_BYTES = 9216
SCALAR_COMPARISON = re.compile(
    r"assert (?:-?[0-9]{1,12}|True|False|None) "
    r"(?:==|!=|<=|>=|<|>) (?:-?[0-9]{1,12}|True|False|None)"
)
SCALAR_ERRORS = (KeyError, IndexError, AttributeError, AssertionError)
SOURCE_LIMIT = 128 * 1024


def _assertions(path, node):
    with path.open("rb") as source:
        raw = source.read(SOURCE_LIMIT + 1)
    if len(raw) > SOURCE_LIMIT:
        return "", ()
    try:
        tree = ast.parse(raw)
    except (ValueError, SyntaxError, RecursionError):
        return "", ()
    qualified = node.split("[", 1)[0].split("::")[1:]
    owner = tree
    for name in qualified:
        owner = next(
            (item for item in owner.body if getattr(item, "name", None) == name), None
        )
        if not isinstance(owner, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return "", ()
    if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "", ()
    nodes = list(owner.body)
    assertions = []
    while nodes:
        item = nodes.pop()
        if isinstance(item, ast.Assert):
            assertions.append((item.lineno, item.end_lineno))
        elif not isinstance(
            item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            nodes.extend(ast.iter_child_nodes(item))
    return hashlib.sha256(raw).hexdigest(), tuple(assertions)


def _run_pytest(node, mock_locals):
    import pytest

    path = Path(node.split("::", 1)[0]).resolve()
    digest, assertions = _assertions(path, node)

    class Probe:
        reports = ()
        observation = None
        traces = None

        @pytest.hookimpl(wrapper=True, tryfirst=True)
        def pytest_runtest_call(self, item):
            function = inspect.unwrap(item.obj)
            code = getattr(function, "__code__", None)
            unsupported = (
                inspect.CO_COROUTINE
                | inspect.CO_ASYNC_GENERATOR
                | inspect.CO_GENERATOR
                | inspect.CO_ITERABLE_COROUTINE
            )
            if (
                mock_locals is None
                or code is None
                or code.co_flags & unsupported
                or Path(code.co_filename).resolve() != path
                or sys.getprofile() is not None
            ):
                return (yield)
            events = 0

            def profile(frame, event, _arg):
                nonlocal events
                events += 1
                if events >= 100_000:
                    sys.setprofile(None)
                elif event == "return" and frame.f_code is code:
                    self.traces = mock_locals(frame)
                    sys.setprofile(None)

            sys.setprofile(profile)
            try:
                return (yield)
            finally:
                sys.setprofile(None)

        def pytest_collection_finish(self, session):
            if [item.nodeid for item in session.items] != [node]:
                pytest.exit("Expected exactly the selected test", returncode=4)

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_makereport(self, item, call):
            report = yield
            if (
                item.nodeid == node
                and report.when == "call"
                and call.excinfo is not None
                and call.excinfo.type in SCALAR_ERRORS
                and (crash := getattr(report.longrepr, "reprcrash", None)) is not None
                and Path(crash.path).resolve() == path
            ):
                line = next(
                    (
                        start
                        for start, stop in assertions
                        if start <= crash.lineno <= stop
                    ),
                    None,
                )
                if line is not None:
                    kind = call.excinfo.type.__name__
                    comparison = ""
                    if kind == "AssertionError":
                        message = (
                            crash.message[:300]
                            .splitlines()[0]
                            .removeprefix("AssertionError: ")
                        )
                        if SCALAR_COMPARISON.fullmatch(message):
                            comparison = message
                    self.observation = {
                        "line": line,
                        "kind": kind,
                        "digest": digest,
                        "comparison": comparison,
                    }
            return report

        def pytest_runtest_logreport(self, report):
            if report.nodeid == node and not hasattr(report, "wasxfail"):
                self.reports += ((report.when, report.outcome),)

    probe = Probe()
    os.environ.pop("PYTEST_ADDOPTS", None)
    sys.path.insert(0, os.getcwd())
    status = pytest.main(
        ["-o", "addopts=", "-p", "no:cacheprovider", "--", node], plugins=[probe]
    )
    expected = "passed" if status == 0 else "failed"
    if status in (0, 1) and probe.reports == (
        ("setup", "passed"),
        ("call", expected),
        ("teardown", "passed"),
    ):
        return int(status), probe.observation, probe.traces
    return 4, None, None


def _main(node, mock_locals):
    with os.fdopen(os.dup(1), "w") as report, open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        status, observation, traces = _run_pytest(node, mock_locals)
        if status in (0, 1):
            result = {"status": status, "observation": observation, "traces": traces}
            payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
            if len(payload) > REPORT_BYTES:
                result["traces"] = None
                payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
            report.write(payload)

        return status


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1], globals().get("_mock_locals")))
