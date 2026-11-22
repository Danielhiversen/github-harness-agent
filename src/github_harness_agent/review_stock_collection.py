"""Fingerprint the executed pytest selection during speculative stock repair."""

import re
from pathlib import PurePosixPath

from .review_prefix_replay import _interpreter

_VERIFY_SELECTION = """
import hashlib
import json
import os
import sys

with os.fdopen(os.dup(1), "w") as result, open(os.devnull, "wb") as sink:
    os.dup2(sink.fileno(), 1)
    os.dup2(sink.fileno(), 2)
    sys.path.insert(0, os.getcwd())
    import pytest

    class Selection:
        nodes = None
        outcomes = {}
        incomplete = False

        @pytest.hookimpl(trylast=True)
        def pytest_collection_finish(self, session):
            if 0 < len(session.items) <= 20000:
                self.nodes = [item.nodeid for item in session.items]

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.nodeid in self.outcomes or getattr(report, "wasxfail", None):
                    self.incomplete = True
                self.outcomes[report.nodeid] = report.outcome
            elif report.when in {"setup", "teardown"} and not report.passed:
                self.incomplete = True

    selection = Selection()
    expected = int(sys.argv[2])
    status = pytest.main(sys.argv[3:], plugins=[selection])
    if (
        status != expected or selection.incomplete or selection.nodes is None
        or sys.argv[1] not in selection.nodes
        or len(selection.nodes) != len(selection.outcomes)
        or set(selection.nodes) != selection.outcomes.keys()
        or any(result not in {"passed", "failed"} for result in selection.outcomes.values())
    ):
        sys.exit(4)
    failures = {name for name, outcome in selection.outcomes.items() if outcome == "failed"}
    if failures != ({sys.argv[1]} if expected else set()):
        sys.exit(4)
    data = json.dumps(selection.nodes, ensure_ascii=True, separators=(",", ":")).encode()
    result.write(f"{len(selection.nodes)} {hashlib.sha256(data).hexdigest()}")
"""


def stock_collection(
    executor, task, archive, budget, patch, command, node, expected, *, failed=False
):
    """Verify the observed pytest selection without returning node IDs or fixture values."""
    interpreter = _interpreter(command)
    if interpreter is None:
        return None
    args = command[len(interpreter) :]
    if args[:2] == ("-m", "pytest"):
        args = args[2:]
    if PurePosixPath(interpreter[0]).name == "uv":
        interpreter = ("env", "UV_OFFLINE=1", *interpreter)
    with executor.verification_runner.workspace(
        archive, budget.remaining, budget.cancelled
    ) as ws:
        tree = executor._stage_baseline(ws, budget)
        if patch:
            executor._git(ws, budget, "apply", "--index", data=patch)
            executor._check_patch_paths(ws, task, budget, tree)
        budget.claim("command", budget.limits.commands)
        outcome = executor._command(
            ws,
            budget,
            (
                *interpreter,
                "-I",
                "-B",
                "-c",
                _VERIFY_SELECTION,
                node,
                str(int(failed)),
                *args,
            ),
            checked=False,
            cap=256,
        )
    match = re.fullmatch(rb"([1-9][0-9]{0,4}) ([a-f0-9]{64})", outcome.stdout)
    if (
        not outcome.returncode
        and not outcome.stderr
        and match
        and int(match[1]) == expected
    ):
        return match[2].decode()
    return None
