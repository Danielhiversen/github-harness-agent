import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from github_harness_agent.check_diagnostics import ordinary_unused_import_failure
from github_harness_agent.review_solver import (
    _PytestRun,
    _RegressionRepair,
    _qualified_stock_operator_cases,
    _retry_stock_repair,
)

PYTEST = (sys.executable, "-m", "pytest", "test_operator.py")
RUFF = ("ruff", "check", "module.py")
MYPY = ("mypy", "module.py")


@pytest.mark.parametrize(
    ("passed", "failed", "dependency_seed", "trusted", "expected"),
    [
        (0, 4, False, True, True),
        (3, 4, True, True, False),
        (10, 5, False, True, False),
        (5, 5, True, True, True),
        (6, 6, True, True, True),
        (4, 5, True, True, False),
        (5, 6, True, True, False),
        (10, 7, True, True, False),
        (10, 5, True, False, False),
    ],
)
def test_dependency_seed_case_exception_requires_trusted_balanced_selection(
    passed, failed, dependency_seed, trusted, expected
):
    tests = [f"test_case_{index}" for index in range(failed)]
    nodes = frozenset(f"test_operator.py::{item}" for item in tests)
    failures = {(0, node) for node in nodes}
    verified_names = tests if trusted else tests[:-1]
    assert (
        _qualified_stock_operator_cases(
            {0: _PytestRun(passed + failed, frozenset(), 0)},
            {0: _PytestRun(passed, nodes, 0)},
            failures,
            (PYTEST,),
            {"test_operator.py": set(verified_names)},
            dependency_seed=dependency_seed,
        )
        is expected
    )


def test_dependency_seed_case_exception_rejects_changed_operator_selection():
    tests = {f"test_case_{index}" for index in range(5)}
    failed = frozenset(f"test_operator.py::{name}" for name in tests)
    assert not _qualified_stock_operator_cases(
        {0: _PytestRun(12, frozenset(), 0)},
        {0: _PytestRun(6, failed, 0)},
        {(0, item) for item in failed},
        (PYTEST,),
        {"test_operator.py": tests},
        dependency_seed=True,
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("module.py:3:1: F401 unused import\nFound 1 error.\n", True),
        ("F401 unused import\n --> module.py:3:1\nFound 1 error.\n", True),
        (
            "module.py:3:1: F401 unused import\nmodule.py:4:2: I001 unsorted\nFound 2 errors.\n",
            False,
        ),
        ("module.py:3:1: F401 unused import\nFound 2 errors.\n", False),
    ],
)
def test_ruff_is_deferred_only_on_complete_unused_import_diagnostics(output, expected):
    assert ordinary_unused_import_failure(RUFF, output) is expected


def test_only_dependency_seed_prioritizes_behavior_ahead_of_unused_imports():
    behavior, ruff, mypy = (
        (0, "test_operator.py::test_case_0"),
        (1, "check 2 (exit 1)"),
        (2, "check 3 (exit 1)"),
    )
    command_order = (PYTEST, RUFF, MYPY)
    kwargs = {
        "origin": (3, "tests/test_stock.py::test_stock"),
        "sources": {},
        "failures": {},
        "gained": set(),
        "introduced": {behavior},
        "stock_origin": True,
        "unused_import_checks": frozenset({1}),
    }
    ordinary = _RegressionRepair(**kwargs, static_failures={ruff: 1})
    pinned = _RegressionRepair(
        **kwargs, static_failures={ruff: 1}, dependency_seed=True
    )
    mixed = _RegressionRepair(
        **kwargs, static_failures={ruff: 1, mypy: 1}, dependency_seed=True
    )
    static_only = _RegressionRepair(
        **{**kwargs, "introduced": set()},
        static_failures={ruff: 1},
        dependency_seed=True,
    )
    assert ordinary.next_target(command_order) == ruff
    assert pinned.next_target(command_order) == behavior
    assert mixed.next_target(command_order) == mypy
    assert static_only.next_target(command_order) == ruff


def test_dependency_retry_rotates_only_after_two_checked_stalls():
    first = (0, "test_operator.py::test_case_0")
    second = (0, "test_operator.py::test_case_1")
    ruff = (1, "check 2 (exit 1)")
    mypy = (2, "check 3 (exit 1)")
    checks = (PYTEST, RUFF, MYPY)
    seed = _RegressionRepair(
        origin=(3, "tests/test_stock.py::test_stock"),
        sources={},
        failures={},
        gained=set(),
        introduced={first, second},
        stock_origin=True,
        dependency_seed=True,
        static_failures={ruff: 1},
        unused_import_checks=frozenset({1}),
    )
    retry = _retry_stock_repair(seed, first, "checked", False, checked=True)
    assert retry.next_target(checks) == first
    final = _retry_stock_repair(retry, first, "checked again", False, checked=True)
    assert final.next_target(checks) == second
    assert replace(final, introduced={first}).next_target(checks) == first
    assert (
        replace(final, static_failures={ruff: 1, mypy: 1}).next_target(checks) == mypy
    )
    assert replace(final, unused_import_checks=frozenset()).next_target(checks) == ruff
    assert (
        replace(final, dependency_seed=False, static_failures={}).next_target(checks)
        == first
    )
    invalid = _retry_stock_repair(retry, first, "malformed", True)
    assert invalid.next_target(checks) == first
    assert _retry_stock_repair(final, second, "exhausted", False, checked=True) is None


@pytest.mark.parametrize(
    ("parent_seconds", "dependency_seed", "expected"),
    [(500, True, 120), (500, False, 90), (150, True, 90), (70, True, 10)],
)
def test_dependency_double_fingerprint_budget_has_its_own_ceiling(
    monkeypatch, parent_seconds, dependency_seed, expected
):
    from github_harness_agent import review_solver

    monkeypatch.setattr(review_solver, "time", SimpleNamespace(monotonic=lambda: 100))
    budget = SimpleNamespace(
        remaining=lambda: parent_seconds,
        cancelled=lambda: False,
        limits=object(),
    )
    probe = review_solver._pinned_restore_budget(
        budget, dependency_seed=dependency_seed
    )
    assert probe.remaining() == expected
