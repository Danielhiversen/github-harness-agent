import os
import subprocess
import sys
from pathlib import Path

import pytest

from github_harness_agent.review_guard_probe import counterfactual_guard_diagnostic

BASELINE = """class Directory:
    def collect(self, values):
        selected = []
        for value in values:
            selected.append(value)
        return selected
"""
DRAFT = BASELINE.replace(
    "            selected.append(value)",
    "            if value:\n                selected.append(value)",
)
STOCK_TEST = """from subject import Directory

def test_collect():
    directory = Directory()
    assert directory.collect([1, 2, 3]) == EXPECTED
    assert directory.collect([1, 2, 3]) == EXPECTED
"""


@pytest.mark.parametrize(
    ("expected", "modes", "finding"),
    [
        ("[2, 3]", 2, "suppressed the first true result per function activation"),
        ("[]", 3, "suppressed every true result"),
        ("[17]", 3, ""),
    ],
)
def test_counterfactual_replays_an_unchanged_stock_test_in_fresh_checkouts(
    tmp_path, expected, modes, finding
):
    stock_test = STOCK_TEST.replace("EXPECTED", expected)
    current = {"subject.py": DRAFT}
    original = {"subject.py": BASELINE}
    runs = []

    def replay(sources, timeout):
        checkout = tmp_path / str(len(runs))
        checkout.mkdir()
        test_path = checkout / "test_stock.py"
        test_path.write_text(stock_test)
        for path, text in sources.items():
            (checkout / path).write_text(text)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "test_stock.py::test_collect",
            ],
            cwd=checkout,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            timeout=timeout,
            capture_output=True,
            check=False,
        )
        assert test_path.read_text() == stock_test
        runs.append(Path(checkout / "subject.py").read_text())
        assert result.returncode in (0, 1), result.stdout.decode(errors="replace")
        return result.returncode

    detail = counterfactual_guard_diagnostic(original, current, stock_test, replay)
    assert len(runs) == modes
    assert runs[0] == DRAFT
    assert current["subject.py"] == DRAFT
    assert original["subject.py"] == BASELINE
    if finding:
        assert "subject.py::Directory.collect" in detail
        assert "line 5" in detail
        assert "if value" in detail
        assert finding in detail
        assert "diagnostic only" in detail
    else:
        assert not detail


def test_counterfactual_does_not_mutate_when_control_run_passes():
    runs = []

    def replay(sources, timeout):
        runs.append(sources["subject.py"])
        return 0

    assert not counterfactual_guard_diagnostic(
        {"subject.py": BASELINE}, {"subject.py": DRAFT}, STOCK_TEST, replay
    )
    assert runs == [DRAFT]


@pytest.mark.parametrize("later_assert", [False, True])
def test_sandbox_observer_reports_partial_advance_without_exception_data(
    tmp_path, monkeypatch, later_assert
):
    import tarfile
    from io import BytesIO

    from test_executor import FixtureRunner

    from github_harness_agent.executor import (
        Limits,
        PlainExecutor,
        TaskSnapshot,
        _Budget,
    )
    from github_harness_agent.review_guard_probe import stock_guard_diagnostic
    from github_harness_agent.review_solver import patch_for

    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    before = """class Directory:
    def collect(self, values, sink):
        found = {}
        for value in values:
            if (
                value > 100
            ):
                continue
            sink.record(value, "hidden-argument")
            found.setdefault("hidden-item", []).append(value)
        return found
"""
    after = before.replace("value > 100", "value > 0")
    if later_assert:
        check = 'assert len(result["hidden-item"]) >= 1\n    assert 7 == 9'
    else:
        check = 'assert len(result["hidden-item"]) == 2'
    stock = (
        "from subject import Directory\n"
        "from unittest.mock import Mock\n"
        "def test_stock():\n"
        "    sink = Mock()\n"
        "    sink.record(10)\n"
        "    sink.record(20)\n"
        "    result = Directory().collect([1, 2, 3], sink)\n"
        f"    {check}\n"
        'def test_unselected():\n    raise RuntimeError("unselected")\n'
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, source in {"subject.py": before, "test_stock.py": stock}.items():
            raw = source.encode()
            item = tarfile.TarInfo(path)
            item.size = len(raw)
            snapshot.addfile(item, BytesIO(raw))
    runner = FixtureRunner(tmp_path)
    executor = PlainExecutor(runner, None, Limits(), verification_runner=runner)
    command = (sys.executable, "-m", "pytest", "-q", "test_stock.py")
    task = TaskSnapshot("example/fixture", "1" * 40, "Exercise guard.", (command,))
    budget = _Budget(executor.limits, lambda: False, task)

    output = stock_guard_diagnostic(
        executor,
        task,
        archive.getvalue(),
        budget,
        command,
        "test_stock.py::test_stock",
        {"subject.py": before},
        {"subject.py": after},
        stock,
        patch_for,
    )

    assert "partial diagnostic only" in output
    assert "still failed" in output
    assert ("later tracked assert" if later_assert else "same tracked assert") in output
    assert ("assert 7 == 9" if later_assert else "assert 1 == 2") in output
    assert "KeyError" in output
    assert "sink call 3" in output
    assert '[1,"<redacted>"]' in output
    assert "hidden-item" not in output
    assert "hidden-argument" not in output
    assert len(runner.workspaces) == 3
    assert all(
        (workspace / "test_stock.py").read_text() == stock
        for workspace in runner.workspaces
    )
