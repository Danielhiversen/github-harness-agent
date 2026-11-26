import sys

import pytest

from github_harness_agent.executor import ExecutionError, LimitExceeded, Limits, PlainExecutor, TaskSnapshot
from github_harness_agent.rebase import BOT_IDENTITY, git
from github_harness_agent.review_solver import patch_for
from github_harness_agent.review_recovery import ReviewRecovery
from github_harness_agent.test_updates import (
    authorized_regression_base, authorized_regression_request, regression_assertion_report, requests_new_tests,
)
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


def test_mixed_regression_must_explicitly_authorize_a_reviewed_production_file():
    sha = "b" * 40
    command = "Update production code and add a new regression test."
    assert requests_new_tests(command)
    item = {
        "key": "comment:1:revision", "author": "maintainer", "thread": "thread", "reply_to": 1,
        "path": "src/value.py", "body": f"{command}\nVerify regression against: {sha}",
    }
    assert authorized_regression_request([item]) == (sha, "src/value.py")
    batch = [
        {"key": "review:2", "author": "maintainer", "body": "Fix the queue."},
        {**item, "key": "comment:3", "body": "Preserve existing region counts."},
        item,
    ]
    assert authorized_regression_request(batch) == (sha, "src/value.py")
    with pytest.raises(ValueError, match="one regression verification"):
        authorized_regression_request([*batch, {**item, "key": "comment:4", "path": "src/other.py"}])
    assert authorized_regression_request([{**item, "body": "> " + item["body"]}]) == (None, None)
    with pytest.raises(ValueError, match="production"):
        authorized_regression_request([{**item, "path": "tests/test_existing.py"}])


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
    "case",
    (
        "current",
        "rebased",
        "tracked_test",
        "other_source",
        "test_only",
        "first_pr",
        "first_pr_vacuous",
        "first_pr_unittest",
        "first_pr_rebased",
    ),
)
def test_mixed_pinned_regression_only_overlays_new_tests_on_original_production(
    tmp_path, case
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-q")
    path, other_path, existing = "fixture_value.py", "other.py", "tests/test_existing.py"
    before = "def answer():\n    return 1\n"
    stock = "def test_smoke():\n    assert True\n"
    if case == "first_pr_unittest":
        stock = "import unittest\nclass Smoke(unittest.TestCase):\n    def test_smoke(self):\n        self.assertTrue(True)\n"
    (checkout / "tests").mkdir()
    (checkout / path).write_text(before)
    (checkout / other_path).write_text("value = 1\n")
    (checkout / existing).write_text(stock)
    parent = commit(checkout, "parent")
    (checkout / "origin.txt").write_text("original PR head\n")
    origin = commit(checkout, "origin")
    base = origin
    if case in {"rebased", "first_pr_rebased"}:
        git(checkout, "checkout", "--detach", parent)
        (checkout / "target.txt").write_text("synthetic rebased target\n")
        base = commit(checkout, "rebased")
    new_path = "tests/test_review_regressions.py"
    test_source = "from fixture_value import answer\n\ndef test_answer():\n    assert answer() == 2\n"
    if case == "first_pr_vacuous":
        test_source = test_source.replace("== 2", ">= 1")
    if case == "first_pr_unittest":
        test_source = "import unittest\nfrom fixture_value import answer\nclass ValueTests(unittest.TestCase):\n    def test_answer(self):\n        self.assertEqual(answer(), 2)\n"
    original = {path: before, new_path: ""}
    patched = {path: before.replace("return 1", "return 2"), new_path: test_source}
    if case == "tracked_test":
        original[existing], patched[existing] = stock, stock.replace("True", "1 == 1")
    if case == "other_source":
        original[other_path], patched[other_path] = "value = 1\n", "value = 2\n"
    patch = patch_for(original, patched)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    runner = FixtureRunner(sandbox)
    task = TaskSnapshot(
        "org/fixture",
        base,
        "Add regression coverage.",
        ((sys.executable, "-m", "unittest", "discover", "-s", "tests"),)
        if case == "first_pr_unittest"
        else ((sys.executable, "-m", "pytest", "-q", existing),),
        require_new_tests=True,
        regression_base=parent
        if case == "test_only"
        else base
        if case == "first_pr_rebased"
        else origin,
        regression_head=None if case == "test_only" else origin,
        regression_edit_path=None
        if case
        in {
            "test_only",
            "first_pr",
            "first_pr_vacuous",
            "first_pr_unittest",
            "first_pr_rebased",
        }
        else path,
        first_pr_regression=case.startswith("first_pr"),
    )
    executor = PlainExecutor(runner, None)
    if case in {"tracked_test", "other_source", "test_only"}:
        with pytest.raises(ExecutionError, match="out-of-scope path"):
            executor.verify_patch(task, checkout, patch)
        return
    if case == "first_pr_vacuous":
        with pytest.raises(ExecutionError, match="tests also pass"):
            executor.verify_patch(task, checkout, patch)
        return
    result = executor.verify_patch(task, checkout, patch)
    assert result.patch == patch
    assert any(check.command[0] == "verify-regression" and not check.returncode for check in result.checks)
    if case in {"first_pr", "first_pr_unittest"}:
        proof = next(check for check in result.checks if check.command[0] == "verify-regression")
        assert "Observed baseline failure:" in proof.output
        assert "test_answer (assertion failure)" in proof.output
        assert "Passing candidate:" in proof.output
    historical = runner.workspaces[-1]
    if case == "first_pr_rebased":
        assert (historical / "target.txt").read_text() == "synthetic rebased target\n"
    assert (historical / path).read_text() == before
    assert (historical / existing).read_text() == stock
    assert (historical / new_path).read_text() == test_source
    assert not git(checkout, "status", "--porcelain")


@pytest.mark.parametrize("case", ("mock-change", "same", "genuine", "budget", "probe-error"))
def test_regression_callback_advice_is_private_and_advisory(tmp_path, monkeypatch, case):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-q")
    source = checkout / "fixture_value.py"
    old = "def dispatch(callback):\n    return callback([1234567, 2])\n"
    source.write_text(old)
    (checkout / "tests").mkdir()
    (checkout / "tests" / "test_existing.py").write_text("def test_smoke():\n    assert True\n")
    ancestor = commit(checkout, "old")
    source.write_text(old.replace("1234567", "7654321") if case != "same" else old + "\n# Current\n")
    base = commit(checkout, "current")
    new_path = "tests/test_review_regressions.py"
    callback = "batch[0]" if case == "genuine" else "len(batch)"
    apply = "Mock(side_effect=fake)" if case == "mock-change" else "fake"
    expected = 7654321 if case == "genuine" else 2
    wanted = (
        "from unittest.mock import Mock\nfrom fixture_value import dispatch\n"
        "def test_dispatch():\n"
        "    def fake(batch):\n"
        f"        return {callback}\n"
        f"    assert dispatch({apply}) == {expected}\n"
    )
    patch = patch_for({new_path: ""}, {new_path: wanted})
    sandbox = tmp_path / "verifier"
    sandbox.mkdir()
    runner = FixtureRunner(sandbox)
    command = (sys.executable, "-m", "pytest", "-q", "tests/test_existing.py")
    task = TaskSnapshot(
        "org/fixture", base, "Add a regression test.", (command,),
        require_new_tests=True, regression_base=ancestor,
        review_recovery=ReviewRecovery(((source.name, 1, "Add a regression test."),), "", test_only=True),
    )
    executor = PlainExecutor(runner, None, Limits(commands=2) if case == "budget" else Limits())
    if case == "probe-error":
        from github_harness_agent import review_mock_replay

        def broken(*args):
            raise RuntimeError("private trace error")

        monkeypatch.setattr(review_mock_replay, "passing_callback_hint", broken)
    if case == "genuine":
        result = executor.verify_patch(task, checkout, patch)
        assert all(check.returncode == 0 for check in result.checks)
        assert len(runner.workspaces) == 4
        return
    with pytest.raises(ExecutionError, match="named assertion") as error:
        executor.verify_patch(task, checkout, patch)
    detail = str(error.value)
    if case in {"budget", "probe-error"}:
        assert "callback" not in detail and "private trace error" not in detail
        assert len(runner.workspaces) == 4
        return
    assert "fake" in detail
    assert "1234567" not in detail and "7654321" not in detail and "def fake" not in detail
    if case == "mock-change":
        assert "arguments first differed at call 1" in detail
    else:
        assert "callback observations match" in detail
    assert len(runner.workspaces) == 6


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
