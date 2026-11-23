import sys

import pytest

from github_harness_agent.executor import ExecutionError, LimitExceeded, Limits, PlainExecutor, TaskSnapshot
from github_harness_agent.rebase import BOT_IDENTITY, git
from github_harness_agent.review_solver import patch_for
from github_harness_agent.test_updates import authorized_regression_base, regression_assertion_report
from test_executor import FixtureRunner


def test_regression_directive_requires_unquoted_authorized_inline_test_only_feedback():
    crc = "a" * 40
    request = (
        "Please add an offline pytest regression. "
        "A test-only change in a new test module is sufficient.\n\n"
    )
    item = {
        "key": "comment:1:revision", "path": "fixture_value.py",
        "author": "maintainer", "thread": "thread", "reply_to": 1,
    }
    line = f"Verify regression against: {crc}"
    assert authorized_regression_base([{**item, "body": request + line}]) == crc
    for example in (
        request + "> " + line, request + f"```text\n{line}\n```", request + f"<!-- {line} -->",
        request + "<!-- hidden\n" + line, "Fix the production code first.\n" + line,
    ):
        assert authorized_regression_base([{**item, "body": example}]) is None
    assert authorized_regression_base([{**item, "key": "review:1", "body": request + line}]) is None
    with pytest.raises(ValueError, match="full commit SHA"):
        authorized_regression_base([{**item, "body": request + "Verify regression against: main"}])


@pytest.mark.parametrize(
    "report",
    (
        '<testcase classname="cases" name="test_value"><error message="AssertionError: fixture setup" /></testcase>',
        '<testcase classname="cases" name="test_value"><failure message="ValueError: AssertionError" /></testcase>',
        '<testcase classname="cases" name="test_other"><failure message="AssertionError: value" /></testcase>',
    ),
)
def test_regression_requires_a_named_call_assertion_in_the_passing_collection(report):
    green = b'<testsuite><testcase classname="cases" name="test_value" /></testsuite>'
    assert not regression_assertion_report(green, f"<testsuite>{report}</testsuite>".encode())


def commit(path, message):
    name, email = BOT_IDENTITY
    git(path, "add", "-A")
    git(path, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-qm", message)
    return git(path, "rev-parse", "HEAD").decode().strip()


@pytest.mark.parametrize(
    "case", ("regression", "ineffectual", "call_error", "collection_error", "non_ancestor", "archive_limit", "rewritten")
)
def test_ancestor_regression_verification(tmp_path, case):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-q")
    source = checkout / "fixture_value.py"
    source.write_text("def answer():\n    return 1\n")
    extra = checkout / "old_asset.txt"
    if case == "archive_limit":
        extra.write_text("x" * 30000)
    (checkout / "tests").mkdir()
    (checkout / "tests" / "test_existing.py").write_text("def test_smoke():\n    assert True\n")
    ancestor = commit(checkout, "old")
    if case == "non_ancestor":
        source.write_text("def answer():\n    return -2\n")
        other = commit(checkout, "sibling")
        git(checkout, "checkout", "--detach", ancestor)
        ancestor = other
    elif case == "archive_limit":
        extra.unlink()
    source.write_text("def answer():\n    return 2\n\ndef fixed_only():\n    return 2\n")
    base = commit(checkout, "current")
    tests = {
        "ineffectual": "from fixture_value import answer\n\ndef test_answer():\n    assert answer() > 0\n",
        "call_error": "import fixture_value\n\ndef test_answer():\n    assert fixture_value.fixed_only() == 2\n",
        "collection_error": "from fixture_value import fixed_only\n\ndef test_answer():\n    assert fixed_only() == 2\n",
        "rewritten": "from fixture_value import answer\n\ndef test_answer():\n    assert answer() == 3\n",
    }
    wanted = tests.get(case, "from fixture_value import answer\n\ndef test_answer():\n    assert answer() == 2\n")
    new_path = "tests/test_review_regressions.py"
    patch = patch_for({new_path: ""}, {new_path: wanted})
    editor, verifier = tmp_path / "editor", tmp_path / "verifier"
    editor.mkdir()
    verifier.mkdir()
    command = (sys.executable, "-m", "pytest", "-q", "tests/test_existing.py")
    checks = (command,)
    if case == "rewritten":
        rewrite = (
            "from pathlib import Path\n"
            f"p = Path({new_path!r})\n"
            "if p.exists(): p.write_text(p.read_text().replace('== 3', '== 2'))\n"
        )
        checks += ((sys.executable, "-c", rewrite),)
    task = TaskSnapshot(
        "org/fixture", base, "Add a new regression test.", checks,
        require_new_tests=True, regression_base=ancestor,
    )
    verification_runner = FixtureRunner(verifier)
    limits = Limits(archive_bytes=15000) if case == "archive_limit" else Limits()
    executor = PlainExecutor(FixtureRunner(editor), None, limits, verification_runner=verification_runner)
    if case == "regression":
        result = executor.verify_patch(task, checkout, patch)
        assert any(check.command[0] == "verify-regression" and not check.returncode for check in result.checks)
        assert (verification_runner.workspaces[-1] / new_path).read_text() == wanted
        assert "return 1" in (verification_runner.workspaces[-1] / source.name).read_text()
        assert not list(editor.iterdir())
    elif case in {"non_ancestor", "archive_limit"}:
        with pytest.raises(
            LimitExceeded if case == "archive_limit" else ExecutionError,
            match="archive limit" if case == "archive_limit" else "ancestor",
        ):
            executor.verify_patch(task, checkout, patch)
        assert not verification_runner.workspaces
    elif case == "rewritten":
        with pytest.raises(ExecutionError, match="test_answer"):
            executor.verify_patch(task, checkout, patch)
        assert (verification_runner.workspaces[-1] / new_path).read_text() == wanted
    else:
        with pytest.raises(ExecutionError, match="named assertion"):
            executor.verify_patch(task, checkout, patch)
    assert git(checkout, "rev-parse", "HEAD").decode().strip() == base
    assert not git(checkout, "status", "--porcelain")
