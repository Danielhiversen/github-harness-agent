from types import SimpleNamespace

from github_harness_agent.check_diagnostics import check_details, check_failure_summary


def test_ruff_multiline_and_concise_locations_survive_without_messages():
    stdout = (
        "RUF001 String contains ambiguous character confidential-diagnostic\n"
        " --> /tmp/private-work/forecast.py:4:17\n"
        "  |\n"
        "4 | value = 'confidential-source'\n"
        "  |                 ^\n"
        "weather/catalog.py:148:22: RUF002 confidential-docstring\n"
        "RUF003 Comment contains ambiguous character confidential-comment\n"
        "   --> weather/catalog.py:169:26\n"
        "    |\n"
        "169 | # confidential-location\n"
    )
    argv = (
        "uv",
        "run",
        "--locked",
        "--project",
        "/tmp/private-project",
        "ruff",
        "check",
    )
    assert check_details(argv, stdout) == (
        "ruff diagnostics: RUF001 forecast.py:4:17, "
        "RUF002 weather/catalog.py:148:22, RUF003 weather/catalog.py:169:26."
    )


def test_mypy_uses_error_codes_and_locations_only():
    stdout = (
        "/tmp/private-project/main.py:10: error: Incompatible confidential value  [assignment]\n"
        "src/module.py:18:7: error: Missing confidential package  [import-not-found]\n"
        "src/module.py:18: note: confidential extra details\n"
    )
    argv = ("uv", "run", "--offline", "--no-sync", "python", "-m", "mypy", "src/")
    assert check_details(argv, stdout) == (
        "mypy diagnostics: [assignment] main.py:10, "
        "[import-not-found] src/module.py:18:7."
    )


def test_pytest_summary_preserves_only_node_identifiers():
    stdout = "FAILED /tmp/private-run/test_example.py::test_after[private-param] - raw private failure"
    argv = (
        "env",
        "PRIVATE_KEY=not-disclosed",
        "uv",
        "run",
        "--no-sync",
        "--",
        "pytest",
        "-q",
    )
    assert check_details(argv, stdout) == "Failing tests: test_example.py::test_after."


def test_unknown_checks_return_only_ordinal_and_exit_status():
    argv = ("env", "PRIVATE_KEY=not-disclosed", "/tmp/private-bin/custom")
    failed = SimpleNamespace(
        command=argv,
        returncode=1,
        output="FAILED test_example.py::test_private - private-output",
    )
    task = SimpleNamespace(checks=(argv,), pull_checks=())
    assert (
        check_failure_summary(task, (failed,))
        == "Required check 1 exited with status 1."
    )
    assert (
        check_details(("uv", "run", "--with", "pytest", "custom"), failed.output) == ""
    )
    assert check_details(("uv", "run", "--script", "pytest"), failed.output) == ""
