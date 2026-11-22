import sys
import tarfile
from pathlib import Path
from dataclasses import replace
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import pytest
from test_executor import FixtureRunner

from github_harness_agent import review_recovery
from github_harness_agent.checkpoint import DraftCheckpoint
from github_harness_agent.executor import (
    AttemptRejected,
    Limits,
    PlainExecutor,
    TaskSnapshot,
    _Budget,
)
from github_harness_agent.review_recovery import ReviewRecovery
from github_harness_agent.review_solver import (
    _GUIDANCE,
    _line_failure_details,
    apply_blocks,
    evidence,
    patch_for,
    solve,
)

SOURCE = "fixture_calc.py"
NEW_TEST = "tests/test_review_regressions.py"
BASE_SOURCE = "def first():\n    return -1\n\ndef second():\n    return -2\n"


def edit(path, old, new):
    return f"{path}\n<<<<<<< SEARCH\n{old}=======\n{new}>>>>>>> REPLACE\n"


class FreeGateway:
    def __init__(self, *proposals):
        self.proposals = iter(proposals)
        self.prompts = []

    def complete(
        self,
        *,
        role,
        messages,
        tools,
        timeout,
        after_provider=None,
        max_output_tokens=None,
    ):
        assert role == "coding"
        assert not tools
        assert max_output_tokens == 3072
        self.prompts.append(messages[-1]["content"])
        return SimpleNamespace(
            content=next(self.proposals),
            tool_calls=(),
            role=role,
            route="stand-in-free",
            model="fixture-coder",
            configured_model="stand-in/free",
            usage={},
        )


@pytest.fixture
def solver_case(tmp_path, monkeypatch):
    operator = tmp_path.resolve() / "operator"
    operator.mkdir()
    acceptance = operator / "test_acceptance.py"
    acceptance.write_text(
        "from fixture_calc import first, second\n"
        "\ndef test_first():\n    assert first() == 1\n"
        "\ndef test_second():\n    assert second() == 2\n"
    )
    monkeypatch.setattr(review_recovery, "OPERATOR_CHECK_ROOT", str(operator))

    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        contents = {
            SOURCE: BASE_SOURCE.encode(),
            "tests/test_existing.py": (
                b"import fixture_calc\n\n"
                b"def test_existing():\n"
                b"    assert not getattr(fixture_calc, 'broken_stock', False)\n"
            ),
        }
        for path, content in contents.items():
            entry = tarfile.TarInfo(path)
            entry.size = len(content)
            snapshot.addfile(entry, BytesIO(content))

    pytest_cmd = (sys.executable, "-m", "pytest", "-q")
    checkpoints = []
    task = TaskSnapshot(
        "org/fixture",
        "1" * 40,
        "Correct both results and add a regression test.",
        ((*pytest_cmd, "tests/test_existing.py"),),
        verify_base_first=True,
        inline_review_threads=2,
        pull_checks=((*pytest_cmd, str(acceptance)),),
        require_new_tests=True,
        review_recovery=ReviewRecovery(
            (
                (SOURCE, 2, "Correct the first result."),
                (SOURCE, 5, "Correct the second result."),
            ),
            "",
        ),
        save_checkpoint=checkpoints.append,
    )
    return task, archive.getvalue(), FixtureRunner(tmp_path), checkpoints


def test_evidence_pairs_line_traceback_locations_with_each_pytest_node(solver_case):
    task, archive, runner, _ = solver_case
    executor = PlainExecutor(runner, FreeGateway())
    budget = _Budget(executor.limits, lambda: False, task)
    command = (*task.pull_checks[0][:-1], "--tb=line", task.pull_checks[0][-1])

    failures = evidence(executor, task, archive, budget, b"", (command,))

    by_test = {key[1].split("::")[-1]: detail for key, detail in failures.items()}
    assert by_test.keys() == {"test_first", "test_second"}
    assert "test_acceptance.py:4: assert -1 == 1" in by_test["test_first"]
    assert "test_acceptance.py:7: assert -2 == 2" in by_test["test_second"]
    assert "assert -2 == 2" not in by_test["test_first"]
    assert "assert -1 == 1" not in by_test["test_second"]


def test_line_traceback_does_not_guess_when_a_location_is_missing():
    output = (
        "===== FAILURES =====\n"
        "E    assert result == 1\n"
        "test_sample.py:4: assert result == 1\n"
        "===== short test summary info =====\n"
        "FAILED test_sample.py::test_first\n"
        "FAILED test_sample.py::test_second\n"
    )
    assert _line_failure_details(output) == {}


def test_apply_blocks_is_atomic_when_a_later_search_is_ambiguous():
    source = {SOURCE: "first = 0\nsecond = 0\nsecond = 0\n", NEW_TEST: ""}
    original = source.copy()
    response = edit(SOURCE, "first = 0\n", "first = 1\n") + edit(
        SOURCE, "second = 0\n", "second = 2\n"
    )
    with pytest.raises(ValueError, match="must match exactly once"):
        apply_blocks(response, source, NEW_TEST)
    assert source == original


@pytest.mark.parametrize(
    ("proposal", "message"),
    [
        ("incomplete edit", "No complete edit blocks"),
        (edit("elsewhere.py", "", "pass\n"), "outside the editable source list"),
        (edit(SOURCE, "", "pass\n"), "Empty SEARCH"),
        (edit(SOURCE, "first = 0\n", "first = (\n"), "invalid Python"),
    ],
)
def test_apply_blocks_rejects_invalid_edits(proposal, message):
    with pytest.raises(ValueError, match=message):
        apply_blocks(proposal, {SOURCE: "first = 0\n", NEW_TEST: ""}, NEW_TEST)


FIRST = edit(SOURCE, "    return -1\n", "    return 1\n")
SECOND = edit(SOURCE, "    return -2\n", "    return 2\n")
TEST = (
    "from fixture_calc import first, second\n\n"
    "def test_results():\n"
    "    assert (first(), second()) == (1, 2)\n"
)


def test_solver_continues_a_compound_assertion_before_checkpointing(solver_case):
    task, archive, runner, checkpoints = solver_case
    Path(task.pull_checks[0][-1]).write_text(TEST)
    followup = edit(
        SOURCE,
        "    return 1\n\ndef second():\n    return -2\n",
        "    return 1\n\ndef second():\n    return 2\n",
    )
    gateway = FreeGateway(FIRST, followup, edit(NEW_TEST, "", TEST))
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=3, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert "assert (1, -2) == (1, 2)" in gateway.prompts[1]
    assert "    return 1\n\ndef second():" in gateway.prompts[1]
    assert len(checkpoints) == 1
    assert b"+    return 1\n" in checkpoints[0].patch
    assert b"+    return 2\n" in checkpoints[0].patch
    assert NEW_TEST.encode() in result.patch


def test_solver_offers_named_definitions_only_after_repeated_malformed_edits(
    solver_case,
):
    task, archive, runner, _ = solver_case
    Path(task.pull_checks[0][-1]).write_text(TEST)
    replacement = (
        f"REPLACE {SOURCE}::first\ndef first():\n    return 1\n"
        f"REPLACE {SOURCE}::second\ndef second():\n    return 2\n"
    )
    gateway = FreeGateway("prose", "still prose", replacement, edit(NEW_TEST, "", TEST))
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=4, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert [
        prompt.endswith("Respond with complete named Python definitions only.")
        for prompt in gateway.prompts
    ] == [False, False, True, False]
    assert b"+    return 1\n" in result.patch
    assert b"+    return 2\n" in result.patch
    assert NEW_TEST.encode() in result.patch


def test_solver_discards_a_continuation_when_another_goal_advances(solver_case):
    task, archive, runner, checkpoints = solver_case
    continuation_started = Event()
    other_goal_saved = Event()

    def save(checkpoint):
        checkpoints.append(checkpoint)
        if b"+    return 2\n" in checkpoint.patch:
            other_goal_saved.set()

    class Gateway:
        def complete(self, *, role, messages, **request):
            prompt = messages[-1]["content"]
            goal = prompt.splitlines()[0]
            if "test_second" in goal:
                assert continuation_started.wait(20)
                proposal = SECOND
            elif "unpublished proposal" in prompt:
                assert "assert 0 == 1" in prompt
                assert "assert -2 == 2" not in prompt
                assert "Correct the second result." not in prompt
                continuation_started.set()
                assert other_goal_saved.wait(20)
                proposal = FIRST + edit(
                    SOURCE, "def first():\n", "continued = True\n\ndef first():\n"
                )
            elif "test_first" in goal and not continuation_started.is_set():
                proposal = edit(SOURCE, "    return -1\n", "    return 0\n")
            elif "test_first" in goal:
                assert "return 0" not in prompt
                proposal = FIRST
            else:
                proposal = edit(NEW_TEST, "", TEST)
            return SimpleNamespace(
                content=proposal,
                tool_calls=(),
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                usage={},
            )

    task = replace(task, save_checkpoint=save)
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=5, concurrent_probes=2)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert b"+    return 2\n" in checkpoints[0].patch
    assert all(b"continued = True" not in draft.patch for draft in checkpoints)
    assert b"continued = True" not in result.patch


def test_solver_keeps_verified_progress_across_free_gateway_proposals(solver_case):
    task, archive, runner, checkpoints = solver_case
    regress_first = edit(SOURCE, "    return 1\n", "    return -1\n")
    gateway = FreeGateway(
        "incomplete edit",
        FIRST,
        SECOND + regress_first,
        SECOND,
        edit(NEW_TEST, "", TEST),
    )
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=5, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert b"+    return 1\n" in result.patch
    assert b"+    return 2\n" in result.patch
    assert NEW_TEST.encode() in result.patch
    assert [
        (
            b"+    return 1\n" in c.patch,
            b"+    return 2\n" in c.patch,
            NEW_TEST.encode() in c.patch,
        )
        for c in checkpoints
    ] == [
        (True, False, False),
        (True, True, False),
    ]
    assert "No complete edit blocks" not in gateway.prompts[1]
    assert "Proposal was discarded" in gateway.prompts[3]
    assert all(check.returncode == 0 for check in result.checks)
    assert any(NEW_TEST in check.command for check in result.checks)


@pytest.mark.parametrize("stale_regression", (False, True))
def test_solver_verifies_another_goal_while_a_provider_is_still_working(
    solver_case, stale_regression
):
    task, archive, runner, checkpoints = solver_case
    first_started = Event()
    second_saved = Event()

    def save(checkpoint):
        checkpoints.append(checkpoint)
        if b"+    return 2\n" in checkpoint.patch:
            second_saved.set()

    task = replace(task, save_checkpoint=save)

    class Gateway:
        def complete(self, *, messages, **request):
            goal = messages[-1]["content"].splitlines()[0]
            if "test_first" in goal:
                initial = not first_started.is_set()
                first_started.set()
                assert second_saved.wait(20), (
                    "First provider waited for verified progress"
                )
                route, content = "slow", FIRST
                if stale_regression and initial:
                    content += edit(
                        SOURCE,
                        "def second():\n",
                        "broken_stock = True\n\ndef second():\n",
                    )
                elif stale_regression:
                    assert "test_existing" in messages[-1]["content"]
            elif "test_second" in goal:
                assert first_started.wait(20)
                route, content = "fast", SECOND
            else:
                route, content = "fast", edit(NEW_TEST, "", TEST)
            return SimpleNamespace(
                role=request["role"],
                route=route,
                model=route,
                configured_model=route,
                content=content,
                usage={"requests": 1},
            )

    expected_routes = (
        ["fast", "slow", "slow", "fast"]
        if stale_regression
        else ["fast", "slow", "fast"]
    )
    executor = PlainExecutor(
        runner,
        Gateway(),
        Limits(model_requests=len(expected_routes), concurrent_probes=2),
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert b"+    return 1\n" not in checkpoints[0].patch
    assert b"+    return 2\n" in checkpoints[0].patch
    assert b"+    return 1\n" in result.patch
    assert b"+    return 2\n" in result.patch
    assert all(b"broken_stock" not in checkpoint.patch for checkpoint in checkpoints)
    assert [route.route for route in result.route_trace] == expected_routes
    assert result.usage == {"requests": len(expected_routes)}
    assert budget.model_requests_remaining() == 0


def test_solver_does_not_keep_a_candidate_rejected_by_final_gate(solver_case):
    task, archive, runner, checkpoints = solver_case
    gateway = FreeGateway(
        FIRST, SECOND, edit(NEW_TEST, "", TEST.replace("(1, 2)", "(0, 0)"))
    )
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=3, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    with pytest.raises(AttemptRejected, match="test_results"):
        solve(executor, task, archive, budget, "fixture-version")

    assert b"+    return 2\n" in checkpoints[-1].patch
    assert NEW_TEST.encode() not in checkpoints[-1].patch


def test_solver_final_verifies_a_complete_checkpoint_without_a_new_request(solver_case):
    task, archive, runner, _ = solver_case
    patch = patch_for(
        {SOURCE: BASE_SOURCE, NEW_TEST: ""},
        {SOURCE: BASE_SOURCE.replace("return -", "return "), NEW_TEST: TEST},
    )
    saved = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch,
        guidance=_GUIDANCE,
        last_editor=("stand-in-free", "stand-in/free"),
    )
    resumed = replace(task, checkpoints={saved.key: saved})
    idle_gateway = FreeGateway()
    executor = PlainExecutor(runner, idle_gateway, Limits(model_requests=1))
    budget = _Budget(executor.limits, lambda: False, resumed)
    budget.claim("model request", executor.limits.model_requests)

    result = solve(executor, resumed, archive, budget, "fixture-version")

    assert result.patch == patch
    assert any(NEW_TEST in check.command for check in result.checks)
    assert not idle_gateway.prompts
