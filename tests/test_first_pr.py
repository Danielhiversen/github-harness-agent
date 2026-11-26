import pytest
import subprocess
import sys

from github_harness_agent.first_pr import pull_summary, reproducible_bug
from github_harness_agent.test_updates import (
    focused_python_tests,
    passed_pytest_report,
    regression_assertion_report,
    regression_proof,
)


@pytest.mark.parametrize(
    "title,body,expected",
    [
        (
            "Fix empty response crash",
            "Calling read with an empty response raises ValueError instead of returning None.",
            True,
        ),
        (
            "Wrong queue ordering",
            "After two pushes the queue returns the wrong item.",
            True,
        ),
        ("Bug: parser fails", "parse('x') fails", True),
        ("Add CSV export", "Expected: support exporting records.", False),
        ("Document error handling", "Explain when parsing fails.", False),
        (
            "Bug report template",
            "### Steps to reproduce\n### Expected\n### Actual",
            False,
        ),
        (
            "Address overflow",
            "### Steps to reproduce\nOpen 5 items\n### Expected behavior\n5 visible items\n### Actual behavior\nOnly 4 visible items",
            True,
        ),
        ("Add button crashes", "Clicking Add crashes the application.", True),
    ],
)
def test_selects_observable_bugs(title, body, expected):
    assert reproducible_bug(title, body) is expected


def test_summary_reports_observed_proof_and_supports_legacy_checks():
    patch = b"--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-return 1\n+return 2\n"
    summary = pull_summary(
        "Fix wrong answer",
        patch,
        [
            {
                "command": ["verify-regression", "a" * 40],
                "returncode": 0,
                "output": "Observed baseline failure: assert 1 == 2\nPassing candidate: 1 passed",
            },
            {"command": ["pytest", "-q"], "returncode": 0, "output": "1 passed"},
        ],
    )
    assert "assert 1 == 2" in summary
    assert "Passing candidate: 1 passed" in summary
    legacy = pull_summary("Fix wrong answer", patch, ["pytest -q (exit 0)"])
    assert "Changed files:" in legacy
    assert "Regression proof" not in legacy


@pytest.mark.parametrize(
    "body",
    [
        "import unittest\nclass Case(unittest.TestCase):\n    def setUp(self):\n        self.assertEqual(1, 2)\n    def test_value(self):\n        self.assertEqual(1, 1)\n",
        "def test_value():\n    assert True\n",
    ],
)
def test_unittest_proof_rejects_setup_failure_and_zero_tests(tmp_path, body):
    (tmp_path / "test_case.py").write_text(body)
    command = focused_python_tests(
        ((sys.executable, "-m", "unittest", "discover"),),
        ("test_case.py",),
        "report.xml",
    )
    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert result.returncode == 1
    report = (tmp_path / "report.xml").read_bytes()
    assert not passed_pytest_report(report)
    green = b'<testsuite><testcase classname="test_case.Case" name="test_value" /></testsuite>'
    assert not regression_assertion_report(green, report)


def test_public_proof_does_not_copy_paths_parameter_values_or_assertion_values():
    green = b'<testsuite><testcase classname="private.path" name="test_answer[secret-value]" /></testsuite>'
    red = b'<testsuite><testcase classname="private.path" name="test_answer[secret-value]"><failure message="AssertionError: secret-token">/private/path traceback</failure></testcase></testsuite>'
    proof = regression_proof(green, red)
    assert "test_answer" in proof
    assert "secret" not in proof
    assert "private" not in proof
