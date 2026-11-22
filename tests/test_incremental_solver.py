import json
import re
import sys
import tarfile
from pathlib import Path
from dataclasses import replace
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from test_executor import FixtureRunner, FixtureWorkspace
from test_providers import KILO, OLLAMA, OPENROUTER, OPENROUTER_SIBLING, deployment, routed

from github_harness_agent import review_recovery
from github_harness_agent.checkpoint import DraftCheckpoint
from github_harness_agent.executor import (
    AttemptRejected,
    Limits,
    PlainExecutor,
    ReviewRoundExhausted,
    TaskSnapshot,
    _Budget,
)
from github_harness_agent.review_assertions import (
    assertion_progress,
    operator_assertions,
)
from github_harness_agent.review_recovery import ReviewRecovery
from github_harness_agent.task_context import compact_focused_review
from github_harness_agent.review_solver import (
    _GUIDANCE,
    _rejected_attempt,
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
        exclude=(),
        after_provider=None,
        max_output_tokens=None,
    ):
        assert role == "coding"
        assert not tools
        named = messages[-1]["content"].endswith(
            "Respond with complete named Python definitions only."
        )
        assert max_output_tokens == (6144 if named else 3072)
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


def test_round_preference_is_used_for_only_the_first_model_request(solver_case):
    task, _, _, _ = solver_case
    task = replace(task, retry_after_provider="previous-provider", provider_offset=12)
    preferred = []

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            preferred.append((kwargs.get("after_provider"), kwargs.pop("provider_offset", 0)))
            return super().complete(**kwargs)

    gateway = Gateway("first", "second", "third")
    budget = _Budget(Limits(model_requests=3), lambda: False, task)
    messages = [{"role": "user", "content": "Review."}]
    budget.complete(gateway, "coding", messages, max_output_tokens=3072)
    budget.complete(gateway, "coding", messages, max_output_tokens=3072)
    budget.complete(
        gateway,
        "coding",
        messages,
        max_output_tokens=3072,
        after_provider="chosen-for-goal",
    )
    assert preferred == [("previous-provider", 12), (None, 0), ("chosen-for-goal", 0)]

    # A fresh strategy is independent of the previous unsuccessful coding route.
    class Roles:
        def complete(self, **request):
            preferred.append((request.get("after_provider"), request.get("provider_offset", 0)))
            return SimpleNamespace(role=request["role"], route="free", model="free", usage={})

    budget = _Budget(Limits(model_requests=2), lambda: False, task)
    for role in ("planning", "coding"):
        budget.complete(Roles(), role, messages)
    assert preferred[-2:] == [(None, 0), (None, 0)]


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


def test_ruff_goal_keeps_search_replace_after_malformed_proposals(
    solver_case, monkeypatch
):
    task, archive, runner, _ = solver_case
    task = replace(task, checks=(("ruff", "check", "."),))
    gateway = FreeGateway("prose", "still prose", "more prose")
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=3, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    monkeypatch.setattr(
        "github_harness_agent.review_solver.evidence",
        lambda *args, pytest_runs=None: {
            (1, "check 2 (exit 1)"): f"{SOURCE}:1:1: I001 Imports unsorted"
        },
    )

    with pytest.raises(AttemptRejected):
        solve(executor, task, archive, budget, "fixture-version")

    assert len(gateway.prompts) == 3
    assert all(
        prompt.endswith("Respond with exact SEARCH/REPLACE blocks only.")
        for prompt in gateway.prompts
    )


def _proposal_router(models, proposals, attempts):
    config = deployment(*models)
    for priority, route in enumerate(reversed(config), start=1):
        route["quality_priority"] = priority
        if route["model"] == OPENROUTER_SIBLING:
            route["url"] = "https://openrouter-free.example.invalid"

    def upstream(request):
        model = json.loads(request.content)["model"]
        assert model != OLLAMA.split("/", 1)[1], "Durable exclusion was ignored"
        attempts.append(model)
        return httpx.Response(
            200,
            json={
                "model": "provider-reported-model",
                "choices": [{"message": {"content": next(proposals)}}],
            },
        )

    return routed(config, upstream)


def test_solver_rotates_exact_models_until_verified_progress(solver_case, monkeypatch):
    task, archive, runner, checkpoints = solver_case
    task = replace(
        task,
        checks=(("ruff", "check", "."), ("ruff", "check", "--select", "F401", ".")),
        excluded_models=(tuple(OLLAMA.split("/", 1)),),
    )
    lint = {(1, "check 2 (exit 1)"): "Imports unsorted"}

    def verify(executor, task, archive, budget, patch, commands, *, pytest_runs=None, static_runs=None):
        if b"+    return 1\n" in patch:
            return lint
        return {**lint, (2, "check 3 (exit 1)"): "Unused import"}

    monkeypatch.setattr("github_harness_agent.review_solver.evidence", verify)
    attempts = []
    stalled = edit(SOURCE, "    return -1\n", "    return -3\n")
    gateway = _proposal_router(
        (OLLAMA, OPENROUTER, OPENROUTER_SIBLING, KILO),
        iter((
            "malformed", "malformed", stalled, "malformed", stalled,
            "malformed", "malformed", "malformed", FIRST, "malformed",
        )),
        attempts,
    )
    executor = PlainExecutor(runner, gateway, Limits(model_requests=10, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)

    with pytest.raises(ReviewRoundExhausted):
        solve(executor, task, archive, budget, "fixture-version")

    assert attempts == [
        model.split("/", 1)[1]
        for model in (
            OPENROUTER, OPENROUTER, KILO, KILO, OPENROUTER, OPENROUTER,
            KILO, KILO, OPENROUTER_SIBLING, OPENROUTER,
        )
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0].last_editor == tuple(OPENROUTER_SIBLING.split("/", 1))


def test_solver_reenables_sole_model_once_when_rotation_exhausts(solver_case, monkeypatch):
    task, archive, runner, _ = solver_case
    task = replace(
        task,
        checks=(("ruff", "check", "."),),
        excluded_models=(tuple(OLLAMA.split("/", 1)),),
    )
    monkeypatch.setattr(
        "github_harness_agent.review_solver.evidence",
        lambda *args, pytest_runs=None: {(1, "check 2 (exit 1)"): "Imports unsorted"},
    )
    attempts = []
    gateway = _proposal_router((OLLAMA, OPENROUTER), iter(["malformed"] * 7), attempts)
    executor = PlainExecutor(runner, gateway, Limits(model_requests=7, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)

    with pytest.raises(AttemptRejected):
        solve(executor, task, archive, budget, "fixture-version")

    assert attempts == [OPENROUTER.split("/", 1)[1]] * 6
    assert budget.model_requests_remaining() == 0


def test_assertion_progress_rejects_regressions_and_untrusted_locations():
    fixture_path = "/operator/test_sample.py"
    assertions = operator_assertions(
        {
            "test_sample.py": "def test_results():\n    assert first()\n    assert second()\n"
        },
        "/operator",
    )
    test = (0, "../operator/test_sample.py::test_results")
    before = {test: f"{fixture_path}:2: assert False"}
    after = {test: f"{fixture_path}:3: assert False"}
    commands = (("pytest", "--tb=line", fixture_path),)
    assert assertion_progress(assertions, commands, test, before, after)
    assert not assertion_progress(assertions, commands, test, after, before)
    assert not assertion_progress(
        assertions, commands, test, before, {**after, (1, "Stock lint"): "Failed"}
    )
    assert not assertion_progress(
        assertions,
        commands,
        test,
        before,
        {test: "/checkout/test_sample.py:3: assert False"},
    )


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


@pytest.mark.parametrize("same_draft", (True, False))
@pytest.mark.parametrize("mode", ("stalled", "malformed"))
def test_solver_restores_stalled_goal_mode_only_on_the_validated_draft(
    solver_case, same_draft, mode
):
    task, archive, runner, checkpoints = solver_case
    Path(task.pull_checks[0][-1]).write_text(TEST)
    states = []
    task = replace(
        task,
        save_solver_recovery=lambda state, durable: states.append((state, durable)),
    )
    first = (
        FreeGateway(FIRST, edit(SOURCE, "    return 1\n", "    return 4\n"))
        if mode == "stalled"
        else FreeGateway("prose", "still prose", "more prose", "another invalid edit")
    )
    executor = PlainExecutor(
        runner,
        first,
        Limits(model_requests=2 if mode == "stalled" else 4, concurrent_probes=1),
    )
    budget = _Budget(executor.limits, lambda: False, task)
    with pytest.raises(AttemptRejected):
        solve(executor, task, archive, budget, "fixture-version")
    assert len(states) == 1
    assert not checkpoints
    state, durable = states[0]
    assert durable
    assert [goal[1:] for goal in state.goals] == [
        (0, 2) if mode == "stalled" else (2, 0)
    ]

    drafts = {}
    if not same_draft:
        changed = patch_for(
            {SOURCE: BASE_SOURCE},
            {SOURCE: BASE_SOURCE.replace("return -1", "return 1")},
        )
        draft = DraftCheckpoint(
            "harness2/base", task.base_sha, changed, guidance=_GUIDANCE
        )
        drafts[draft.key] = draft
    resumed = replace(
        task, checkpoints=drafts, solver_recovery=type(state).from_data(state.to_data())
    )
    roles = []
    named_definitions = mode == "malformed" and same_draft
    fix = FIRST + SECOND if same_draft else SECOND
    if named_definitions:
        fix = (
            f"REPLACE {SOURCE}::first\ndef first():\n    return 1\n"
            f"REPLACE {SOURCE}::second\ndef second():\n    return 2\n"
        )
    proposals = iter((fix, edit(NEW_TEST, "", TEST)))

    class Gateway:
        def complete(self, *, role, messages, **request):
            roles.append(role)
            if role == "coding" and roles.count("coding") == 1:
                assert (
                    messages[-1]["content"].endswith(
                        "Respond with complete named Python definitions only."
                    )
                    == named_definitions
                )
            content = (
                '{"strategy": "Fix both outputs.", "symbols": ["fixture_calc.py::first"]}'
                if role == "planning"
                else next(proposals)
            )
            return SimpleNamespace(
                content=content,
                tool_calls=(),
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                usage={},
            )

    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=3, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, resumed)
    result = solve(executor, resumed, archive, budget, "fixture-version")
    assert roles == (
        ["planning", "coding", "coding"]
        if same_draft and mode == "stalled"
        else ["coding", "coding"]
    )
    assert all(check.returncode == 0 for check in result.checks)
    assert len(states) == 2
    assert states[-1] == (None, False)


@pytest.mark.parametrize(
    "planner_reply",
    (
        '{"strategy": "Preserve both results and correct the second return value.", "symbols": ["fixture_calc.py::second"]}',
        '{"strategy": "Read elsewhere", "symbols": ["unrelated.py::missing"]}',
    ),
)
def test_stalled_implementation_uses_bounded_advice_then_normal_verification(
    solver_case, planner_reply
):
    task, archive, runner, checkpoints = solver_case
    Path(task.pull_checks[0][-1]).write_text(TEST)
    prompts = []
    coding = iter(
        (
            FIRST,
            edit(SOURCE, "    return 1\n", "    return 4\n"),
            FIRST + SECOND,
            edit(NEW_TEST, "", TEST),
        )
    )

    class Gateway:
        def complete(self, *, role, messages, timeout, **request):
            prompt = messages[-1]["content"]
            prompts.append((role, prompt))
            if role == "planning":
                assert timeout <= 60
                assert request["max_output_tokens"] == 4096
                assert "fixture_calc.py::second" in prompt
                assert "def test_results():" in prompt
                assert (
                    "def second():"
                    not in prompt.split("Qualified-symbol inventory (quoted):", 1)[1]
                )
                content = planner_reply
            else:
                content = next(coding)
            return SimpleNamespace(
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                content=content,
                usage={},
            )

    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=5, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert [role for role, _ in prompts] == [
        "coding",
        "coding",
        "planning",
        "coding",
        "coding",
    ]
    coding_prompt = prompts[3][1]
    if "unrelated.py" not in planner_reply:
        assert (
            "Preserve both results and correct the second return value."
            in coding_prompt
        )
        assert (
            "fixture_calc.py::second\ndef second():\n    return -2\n" in coding_prompt
        )
    else:
        assert "Read elsewhere" not in coding_prompt
    assert all(b"+    return 4\n" not in draft.patch for draft in checkpoints)
    assert all(check.returncode == 0 for check in result.checks)
    assert budget.model_requests_remaining() == 0


@pytest.mark.parametrize("failure", ("malformed", "same_assertion"))
def test_stalled_goal_reuses_advice_in_a_fresh_coding_prompt(solver_case, failure):
    task, archive, runner, checkpoints = solver_case
    Path(task.pull_checks[0][-1]).write_text(TEST)
    prompts = []
    stalled_edit = edit(SOURCE, "    return -1\n", "    return 4\n")
    malformed = failure == "malformed"
    early_edits = ("An invalid proposal",) if malformed else ()
    guided_failure = "An incomplete edit" if malformed else stalled_edit
    fix = (
        f"REPLACE {SOURCE}::first\ndef first():\n    return 1\n"
        f"REPLACE {SOURCE}::second\ndef second():\n    return 2\n"
        if malformed
        else FIRST + SECOND
    )
    proposals = iter(
        (
            *early_edits,
            FIRST,
            edit(SOURCE, "    return 1\n", "    return 4\n"),
            guided_failure,
            fix,
            edit(NEW_TEST, "", TEST),
        )
    )

    class Gateway:
        def complete(self, *, role, messages, **request):
            prompts.append((role, messages[-1]["content"]))
            content = (
                '{"strategy": "Correct the returns together.", "symbols": ["fixture_calc.py::second"]}'
                if role == "planning"
                else next(proposals)
            )
            return SimpleNamespace(
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                content=content,
                usage={},
            )

    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=6 + len(early_edits), concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert [role for role, _ in prompts] == [
        *(["coding"] * len(early_edits)),
        "coding",
        "coding",
        "planning",
        "coding",
        "coding",
        "coding",
    ]
    assert "Correct the returns together." in prompts[-3][1]
    assert prompts[-3][1].endswith("Respond with exact SEARCH/REPLACE blocks only.")
    retry = prompts[-2][1]
    assert "Correct the returns together." in retry
    assert "fixture_calc.py::second\ndef second():\n    return -2\n" in retry
    assert "Failure details:" in retry
    assert "This checkout includes one unpublished proposal" not in retry
    if malformed:
        assert "Last proposal feedback for this goal:" in retry
        assert retry.endswith("Respond with complete named Python definitions only.")
    else:
        assert "Rejected edit excerpt (quoted data)" in retry
    assert "Correct the returns together." not in prompts[-1][1]
    assert all(b"+    return 4\n" not in draft.patch for draft in checkpoints)
    assert all(check.returncode == 0 for check in result.checks)
    assert budget.model_requests_remaining() == 0


def test_stock_pytest_goal_includes_the_tracked_test_without_edit_access(solver_case):
    task, archive, runner, _ = solver_case
    Path(task.pull_checks[0][-1]).write_text(
        TEST + "\ndef unrelated_operator_helper():\n    return 'operator only'\n"
    )
    stock_regression = edit(SOURCE, "def first():\n", "broken_stock = True\n\ndef first():\n")
    stock_fix = edit(SOURCE, "broken_stock = True\n\n", "")
    gateway = FreeGateway(FIRST + SECOND + stock_regression, stock_fix, edit(NEW_TEST, "", TEST))
    executor = PlainExecutor(runner, gateway, Limits(model_requests=3, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    prompt = gateway.prompts[1]
    assert "Selected failure: tests/test_existing.py::test_existing" in prompt
    assert "def test_existing():" in prompt
    assert "assert not getattr(fixture_calc, 'broken_stock', False)" in prompt
    assert "unrelated_operator_helper" not in prompt
    assert f"Editable checkout paths (use these exact names): {SOURCE}\n" in prompt
    assert b"tests/test_existing.py" not in result.patch
    assert all(check.returncode == 0 for check in result.checks)


def test_review_keeps_an_evaluable_comparison_for_the_next_free_edit(solver_case):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\ndef result():\n    return {}\n",
            "tests/test_existing.py": (
                "from fixture_calc import result\n\n"
                "def test_contract():\n    assert result()['count'] == 1\n"
            ),
        }.items():
            data = content.encode()
            item = tarfile.TarInfo(path)
            item.size = len(data)
            snapshot.addfile(item, BytesIO(data))
    gateway = FreeGateway(
        FIRST + SECOND,
        edit(SOURCE, "    return {}\n", "    return {'count': 2}\n"),
        edit(SOURCE, "    return {'count': 2}\n", "    return {'count': 1}\n"),
    )
    executor = PlainExecutor(runner, gateway, Limits(model_requests=3, concurrent_probes=1))
    result = solve(executor, task, archive.getvalue(), _Budget(executor.limits, lambda: False, task), "fixture")
    assert any(b"return {'count': 2}" in checkpoint.patch for checkpoint in checkpoints)
    assert all(check.returncode == 0 for check in result.checks)


def test_stock_repair_can_finish_two_dependent_edits_privately(solver_case, monkeypatch):
    task, archive, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    mark = edit(SOURCE, "def first():\n", "broken_stock = True\n\ndef first():\n")
    tentative = "enabled = False\nbroken_stock = not enabled\n"
    def trace(*args):
        return "Observed the tentative call sequence." if b"+enabled = False" in args[-1] else "Observed the saved call sequence."

    monkeypatch.setattr("github_harness_agent.review_solver.mock_call_diagnostic", trace)
    monkeypatch.setattr("github_harness_agent.review_solver.stock_guard_diagnostic", lambda *args: "")
    monkeypatch.setattr("github_harness_agent.review_solver.request_coding_advice", lambda *args: None)
    gateway = FreeGateway(
        FIRST + SECOND + mark,
        edit(SOURCE, "broken_stock = True\n", "broken_stock = 1\n"),
        edit(SOURCE, "broken_stock = True\n", "broken_stock = bool(1)\n"),
        edit(SOURCE, "broken_stock = True\n", tentative),
        edit(SOURCE, "enabled = False\n", "enabled = True\n"),
    )
    executor = PlainExecutor(runner, gateway, Limits(model_requests=8, concurrent_probes=1))
    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture")
    assert "one unpublished proposal" in gateway.prompts[-1]
    assert "Observed the tentative call sequence." in gateway.prompts[-1]
    assert b"+enabled = True\n" in result.patch
    assert not any(b"+enabled = False" in checkpoint.patch for checkpoint in checkpoints)
    assert all(check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize("case", ("hedge", "continuation", "single_slot", "no_budget", "unsaved"))
def test_delayed_stock_hedge_respects_durable_recovery_and_capacity(solver_case, monkeypatch, case):
    from concurrent.futures import ThreadPoolExecutor

    from github_harness_agent.checkpoint import SolverRecovery
    from github_harness_agent.review_solver import _goal_digest

    task, _, runner, _ = solver_case
    hedged = case in {"hedge", "continuation"}
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first\n\ndef test_first():\n    assert first() == -1\n"
    )
    digest = "4" * 64
    target = (1, "tests/test_existing.py::test_stock")
    monkeypatch.setattr("github_harness_agent.review_solver._recovery_digest", lambda *args: digest)
    clock = [0.0]
    monkeypatch.setattr(
        "github_harness_agent.review_solver.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    task = replace(
        task,
        solver_recovery=(
            None if case == "unsaved" else SolverRecovery(digest, ((_goal_digest(target), 0, 2),))
        ),
        require_new_tests=hedged,
    )
    source = BASE_SOURCE + "\ndef stock():\n    return -3\n"
    if case == "continuation":
        accepted = DraftCheckpoint(
            "harness2/base", task.base_sha,
            patch_for({SOURCE: source}, {SOURCE: "# Accepted draft\n" + source}),
            guidance=_GUIDANCE,
        )
        task = replace(task, checkpoints={accepted.key: accepted})
        monkeypatch.setattr("github_harness_agent.review_solver.mock_call_diagnostic", lambda *args: "Mock evidence")
        monkeypatch.setattr("github_harness_agent.review_solver.stock_guard_diagnostic", lambda *args: "")
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: source,
            "tests/test_existing.py": "from fixture_calc import stock\n\ndef test_stock():\n    assert stock() == 3\n",
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    primary_started, alternate_started, release_primary = Event(), Event(), Event()
    stock_requests = []

    class Gateway:
        def complete(self, *, role, messages, after_provider=None, **request):
            if role == "planning":
                route, content = "planner", "No usable strategy"
            elif messages[-1]["content"].startswith("Selected failure: new regression tests\n"):
                route, content = "test-author", edit(
                    NEW_TEST, "", "from fixture_calc import stock\n\ndef test_stock_fix():\n    assert stock() == 3\n"
                )
            else:
                stock_requests.append((messages, after_provider))
                count = len(stock_requests)
                first = 1 if case == "unsaved" else 2
                old = "    return -4\n" if case == "continuation" else "    return -3\n"
                fix = edit(SOURCE, old, "    return 3\n")
                if count < first:
                    route, content = "stalled", edit(SOURCE, "    return -3\n", "    return -4\n")
                elif count == first:
                    primary_started.set()
                    assert release_primary.wait(35)
                    route, content = "primary", "Malformed late reply" if hedged else fix
                else:
                    alternate_started.set()
                    route, content = "alternate", fix
            return SimpleNamespace(
                role=role, route=route, model=route, configured_model=route, content=content,
                usage={"requests": 1}, tool_calls=(),
            )

    requests = {"hedge": 5, "continuation": 5, "single_slot": 4, "no_budget": 3, "unsaved": 2}[case]
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=requests, concurrent_probes=1 if case == "single_slot" else 2)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(solve, executor, task, archive.getvalue(), budget, "fixture-version")
        try:
            assert primary_started.wait(30)
            clock[0] = 59.0
            assert not alternate_started.wait(0.6)
            clock[0] = 60.0
            if hedged:
                assert alternate_started.wait(8)
            else:
                assert not alternate_started.wait(1)
                release_primary.set()
            result = future.result(timeout=30)
        finally:
            release_primary.set()

    assert b"+    return 3\n" in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    if hedged:
        assert stock_requests[-2:] == [(stock_requests[-1][0], "stalled")] * 2
        assert [entry.route for entry in result.route_trace] == ["planner", "stalled", "alternate", "test-author"]
        assert result.usage == {"requests": 4}
        assert NEW_TEST.encode() in result.patch
        assert budget.model_requests_remaining() == 0
        if case == "continuation":
            assert "This checkout includes one unpublished proposal" in stock_requests[-1][0][-1]["content"]
            assert b"+# Accepted draft\n" in result.patch
    else:
        assert not alternate_started.is_set()


def test_stalled_stock_pytest_uses_advice_and_named_fallback(solver_case):
    task, _, runner, _ = solver_case
    task = replace(task, require_new_tests=False)
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\ndef stock_result():\n    return -3\n",
            "tests/test_existing.py": (
                "from fixture_calc import stock_result\n\n"
                "def test_stock_result():\n    assert stock_result() == 3\n"
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    calls = []
    edits = iter((
        FIRST + SECOND,
        edit(SOURCE, "    return -3\n", "    return -4\n"),
        edit(SOURCE, "    return -3\n", "    return -5\n"),
        "An incomplete edit",
        edit(SOURCE, "    return -3\n", "    return -6\n"),
        edit(SOURCE, "    return -3\n", "    return -7\n"),
    ))

    class Gateway:
        def complete(self, *, role, messages, max_output_tokens, **request):
            prompt = messages[-1]["content"]
            calls.append((role, prompt, max_output_tokens))
            if role == "planning":
                content = '{"strategy": "Correct the stock return value.", "symbols": ["fixture_calc.py::stock_result"]}'
            elif prompt.endswith("Respond with complete named Python definitions only."):
                content = f"REPLACE {SOURCE}::stock_result\ndef stock_result():\n    return 3\n"
            else:
                content = next(edits, edit(SOURCE, "    return -3\n", "    return 3\n"))
            return SimpleNamespace(
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                content=content,
                usage={},
            )

    executor = PlainExecutor(runner, Gateway(), Limits(model_requests=8, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive.getvalue(), budget, "fixture-version")

    assert [role for role, _, _ in calls] == ["coding"] * 3 + ["planning"] + ["coding"] * 4
    planner_prompt = calls[3][1]
    assert "def test_stock_result():" in planner_prompt
    assert "assert stock_result() == 3" in planner_prompt
    assert f"Editable checkout paths (use these exact names): {SOURCE}\n" in planner_prompt
    assert "def test_first():" not in planner_prompt
    assert "Correct the stock return value." in calls[4][1]
    assert "Correct the stock return value." in calls[5][1]
    assert "Correct the stock return value." not in calls[6][1]
    assert calls[7][1].endswith("Respond with complete named Python definitions only.")
    assert calls[7][2] == 6144
    assert all(check.returncode == 0 for check in result.checks)
    assert b"tests/test_existing.py" not in result.patch


def test_planned_retry_shares_pinned_mock_call_difference(solver_case):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    stock = "\ndef stock_publish(callback):\n    callback(1)\n"
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, text in {
            SOURCE: BASE_SOURCE + stock,
            "tests/test_existing.py": (
                "from fixture_calc import stock_publish\n"
                "from unittest.mock import Mock\n\n"
                "def test_publish():\n"
                "    callback = Mock()\n"
                "    stock_publish(callback)\n"
                "    callback.assert_called_once_with(1)\n"
            ),
        }.items():
            data = text.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    proposed = iter((
        FIRST + SECOND + edit(SOURCE, "    callback(1)\n", "    callback(2)\n"),
        edit(SOURCE, "    callback(2)\n", "    callback(3)\n"),
        edit(SOURCE, "    callback(2)\n", "    callback(4)\n"),
        edit(SOURCE, "    callback(2)\n", "    callback(5)\n"),
        edit(SOURCE, "    callback(2)\n", "    callback(1)\n"),
    ))
    prompts = []

    class Gateway:
        def complete(self, *, role, messages, **request):
            prompts.append((role, messages[-1]["content"]))
            content = (
                '{"strategy":"Preserve the original callback value.",'
                '"symbols":["fixture_calc.py::stock_publish"]}'
                if role == "planning" else next(proposed)
            )
            return SimpleNamespace(
                role=role, route="stand-in-free", model="fixture-coder",
                configured_model="stand-in/free", content=content, usage={},
            )

    executor = PlainExecutor(runner, Gateway(), Limits(model_requests=6, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive.getvalue(), budget, "fixture-version")

    expected = 'callback on base [["",[1],{}]]\ncallback on current [["",[2],{}]]'
    assert [role for role, _ in prompts] == ["coding"] * 3 + ["planning", "coding", "coding"]
    assert all(expected not in prompt for _, prompt in prompts[:3])
    assert all(expected in prompt for _, prompt in prompts[3:])
    assert all(b"Auxiliary mock calls" not in draft.patch for draft in checkpoints)
    assert all(check.returncode == 0 for check in result.checks)


def test_rejected_edit_evidence_reaches_coder_and_planner_without_persistence(
    solver_case,
):
    task, archive, runner, checkpoints = solver_case
    Path(task.pull_checks[0][-1]).write_text(TEST)
    prompts = []
    broken = edit(SOURCE, "def first():\n", "broken_stock = True\n\ndef first():\n")
    changed = edit(SOURCE, "    return -1\n", "    return 0\n")
    coding = iter((broken, broken + changed, FIRST + SECOND, edit(NEW_TEST, "", TEST)))

    class Gateway:
        def complete(self, *, role, messages, **request):
            prompts.append((role, messages))
            content = (
                '{"strategy": "Fix the result without a global flag.", "symbols": ["fixture_calc.py::first"]}'
                if role == "planning"
                else next(coding)
            )
            return SimpleNamespace(
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                content=content,
                usage={},
            )

    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=5, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive, budget, "fixture-version")

    assert [role for role, _ in prompts] == [
        "coding",
        "coding",
        "planning",
        "coding",
        "coding",
    ]
    for _, messages in prompts[1:4]:
        prompt = messages[-1]["content"]
        assert "Rejected edit excerpt (quoted data)" in prompt
        assert "Trace a concrete fixture counterexample" in prompt
        assert "choose a different strategy" in prompt
        assert "current checkout text for edits" in prompt
        assert "+broken_stock = True" in prompt
        assert "test_existing" in prompt
        assert "test_results" in prompt
    for _, messages in prompts[2:4]:
        prompt = messages[-1]["content"]
        assert "-    return -1" in prompt
        assert "+    return 0" in prompt
    _, coding_messages = prompts[3]
    large = [
        coding_messages[0],
        {
            **coding_messages[1],
            "content": coding_messages[1]["content"].replace(
                "\nCurrent checkout:\n",
                "\nCurrent checkout:\n" + "supplemental context\n" * 2000,
            ),
        },
    ]
    compacted = compact_focused_review(large, max_bytes=12 * 1024)
    assert "+broken_stock = True" in compacted[-1]["content"]
    assert "Rejected edit excerpt" not in prompts[-1][1][-1]["content"]
    assert all(b"broken_stock" not in draft.patch for draft in checkpoints)
    assert b"broken_stock" not in result.patch
    assert budget.model_requests_remaining() == 0


def test_rejection_preserves_both_ends_of_long_diagnostics():
    target = (0, "tests/test_sample.py::test_goal")
    regression = (1, "tests/test_stock.py::test_stock")
    omitted = "context " * 800
    evidence = _rejected_attempt(
        {SOURCE: "first = 0\n"},
        {SOURCE: "first = 1\n"},
        target,
        {target: "previous assertion"},
        {
            target: f"Assertion at start {omitted} assertion at end",
            regression: f"Check location {omitted} stock tail",
        },
    )
    assert "Assertion at start" in evidence and "assertion at end" in evidence
    assert "Check location" in evidence and "stock tail" in evidence
    assert "-first = 0\n+first = 1\n" in evidence
    assert len(evidence) < 2000


@pytest.mark.parametrize(
    "followup", ("repair", "assertion", "stagnant", "tradeoff", "repeat")
)
def test_private_operator_regressions_are_repaired_without_trading_passing_cases(
    solver_case, followup
):
    task, archive, runner, checkpoints = solver_case
    fixture = Path(task.pull_checks[0][-1])
    fixture.write_text(
        "import fixture_calc as calc\n\n"
        "def test_first():\n    assert calc.first() == 1\n\n"
        "def test_guard_one():\n"
        + ("    assert calc.second() < 0\n" if followup == "assertion" else "")
        + "    assert calc.second() == -2\n\n"
        "def test_guard_two():\n    assert not getattr(calc, 'regression_flag', False)\n"
    )
    task = replace(
        task, pull_checks=((*task.pull_checks[0][:-1], "--tb=line", str(fixture)),)
    )
    flag = edit(SOURCE, "def first():\n", "regression_flag = True\n\ndef first():\n")
    revert_second = edit(SOURCE, "    return 2\n", "    return -2\n")
    remove_flag = edit(SOURCE, "regression_flag = True\n\n", "")
    if followup == "repair":
        repairs = (revert_second, remove_flag)
    elif followup == "assertion":
        repairs = (
            edit(SOURCE, "    return 2\n", "    return -3\n"),
            edit(SOURCE, "    return -3\n", "    return -2\n") + remove_flag,
        )
    elif followup in {"stagnant", "repeat"}:
        repairs = (
            edit(SOURCE, "regression_flag = True\n", "regression_flag = bool(1)\n"),
            FIRST,
        )
        if followup == "repeat":
            repairs = (repairs[0], FIRST + SECOND + flag, FIRST)
    else:
        repairs = (
            revert_second + edit(SOURCE, "    return 1\n", "    return -1\n"),
            FIRST,
        )
    gateway = FreeGateway(
        FIRST + SECOND + flag,
        *repairs,
        edit(NEW_TEST, "", TEST.replace("(1, 2)", "(1, -2)")),
    )
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=len(repairs) + 2, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    targets = [prompt.splitlines()[0].rsplit("::", 1)[-1] for prompt in gateway.prompts]
    assert targets[:3] == [
        "test_first",
        "test_guard_one",
        {"repair": "test_guard_two", "assertion": "test_guard_one"}.get(
            followup, "test_first"
        ),
    ]
    if followup == "repeat":
        assert targets[3] == "test_first"
    assert "assert calc.second() == -2" in gateway.prompts[1]
    assert "+regression_flag = True" in gateway.prompts[1]
    assert "def test_guard_one():" in gateway.prompts[1]
    assert len(checkpoints) == 1
    for patch in [result.patch, checkpoints[0].patch]:
        assert b"regression_flag" not in patch
        assert b"+    return 2\n" not in patch
        assert b"+    return 1\n" in patch
    assert all(check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize(
    "scenario",
    ("compatible", "two_sites", "identical", "canary", "unrelated", "low_budget"),
)
def test_guarded_stock_reconciliation_checks_every_suite(
    solver_case, monkeypatch, scenario
):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    operator_call = "stock()" if scenario == "identical" else "stock(legacy=True)"
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, stock\n\n"
        f"def test_previous_behavior():\n    assert {operator_call} == -2\n\n"
        "def test_unrelated_behavior():\n    assert first() == -1\n"
    )
    canary = (
        "\ndef test_matching_modes():\n    assert stock() == stock(legacy=True)\n"
        if scenario == "canary"
        else ""
    )
    archive = BytesIO()
    gap = (
        "    # keep the shared implementation\n" * 10 if scenario == "two_sites" else ""
    )
    expected_stock = 5 if scenario == "two_sites" else 4
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE
            + "\ndef stock(*, legacy: bool = False):\n    shared = 1\n"
            + gap
            + "    return shared - 3\n",
            "tests/test_existing.py": (
                "from fixture_calc import stock\n\n"
                f"def test_stock_goal():\n    assert stock() == {expected_stock}\n"
                + canary
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    proposal = edit(SOURCE, "    return shared - 3\n", "    return shared + 3\n")
    if scenario == "two_sites":
        proposal += edit(SOURCE, "    shared = 1\n", "    shared = 2\n")
    if scenario == "unrelated":
        proposal += SECOND
    gateway = FreeGateway(proposal, "An incomplete edit")
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=2, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    if scenario == "low_budget":
        monkeypatch.setattr(
            "github_harness_agent.review_solver._GUARD_MIN_REMAINING",
            executor.limits.wall_time + 1,
        )
    guard_failures = []

    def record(*args, **kwargs):
        failures = evidence(*args, **kwargs)
        if b" if legacy else " in args[4] or b"+    if legacy:" in args[4]:
            guard_failures.append({node for _, node in failures})
        return failures

    monkeypatch.setattr("github_harness_agent.review_solver.evidence", record)
    if scenario in {"compatible", "two_sites"}:
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert b"shared - 3 if legacy else shared + 3" in result.patch
        if scenario == "two_sites":
            assert b"+    shared = 1 if legacy else 2\n" in result.patch
        else:
            assert b"+    shared =" not in result.patch
        assert result.route_trace[-1].configured_model == "stand-in/free"
        assert len(gateway.prompts) == 1
        assert guard_failures[-1] == set()
    else:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert "Private regression repair" in gateway.prompts[1]
        assert len(guard_failures) == (2 if scenario in {"identical", "canary"} else 0)
        if scenario == "canary":
            assert any(
                any(node.endswith("::test_matching_modes") for node in failures)
                for failures in guard_failures
            )
    assert not checkpoints


@pytest.mark.parametrize(
    ("signature", "context", "arguments"),
    (
        ("mode: str | None = None", "", (None, "legacy")),
        ("mode", "    if mode == 'legacy':\n        pass\n", ("legacy", "plain")),
    ),
)
def test_guard_uses_a_public_parameter_signature_or_existing_condition(
    signature, context, arguments
):
    from github_harness_agent.review_guards import guarded_sources

    before = f"def select({signature}):\n{context}    return -1\n"
    after = before.replace("return -1", "return 1")
    outcomes = set()
    for candidate in guarded_sources(before, after):
        assert candidate.startswith(f"def select({signature}):\n{context}")
        namespace = {}
        exec(candidate, namespace)
        outcomes.add(tuple(namespace["select"](arg) for arg in arguments))
    assert outcomes == {(1, -1), (-1, 1)}


@pytest.mark.parametrize(
    "scenario", ("revert", "six_hunks", "required_hunk", "low_budget")
)
def test_selective_stock_revert_checks_both_suites(solver_case, monkeypatch, scenario):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, guard\n\n"
        "def test_guard():\n    assert guard() == 0\n\n"
        "def test_first():\n    assert first() == -1\n"
    )
    stock_guard = (
        "\ndef test_required_guard():\n    with pytest.raises(ValueError):\n        guard()\n"
        if scenario == "required_hunk"
        else ""
    )
    extra = (30, 40, 50) if scenario == "six_hunks" else ()
    extra_source = "".join(
        f"\ndef extra_{value}():\n    return -{value}\n" for value in extra
    )
    extra_tests = "".join(
        f"\ndef test_extra_{value}():\n    from fixture_calc import extra_{value}\n    assert extra_{value}() == {value}\n"
        for value in extra
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE
            + "\ndef left():\n    return -10\n\ndef right():\n    return -20\n"
            + extra_source
            + "\ndef guard():\n    return 0\n",
            "tests/test_existing.py": (
                "import pytest\nfrom fixture_calc import left, right, guard\n\n"
                "def test_stock():\n    assert (left(), right()) == (10, 20)\n"
                + stock_guard
                + extra_tests
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    left_edit = (
        "    # Preserve the stock slice main value.\n    return 10\n"
        if extra
        else "    return 10\n"
    )
    proposal = (
        edit(SOURCE, "    return -10\n", left_edit)
        + edit(SOURCE, "    return -20\n", "    return 20\n")
        + edit(SOURCE, "    return 0\n", "    raise ValueError('guard regression')\n")
        + "".join(
            edit(SOURCE, f"    return -{value}\n", f"    return {value}\n")
            for value in extra
        )
    )
    gateway = FreeGateway(proposal, "An incomplete edit")
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=2, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    if scenario == "low_budget":
        monkeypatch.setattr(
            "github_harness_agent.review_solver._GUARD_MIN_REMAINING",
            executor.limits.wall_time + 1,
        )

    probes = []
    needed = tuple(f"+    return {value}\n".encode() for value in (10, 20, *extra))
    rejected = b"+    raise ValueError"
    additions = (*needed, rejected)

    def record(*args, **kwargs):
        failures = evidence(*args, **kwargs)
        patch = args[4]
        if sum(line in patch for line in additions) == len(needed):
            probes.append((patch, failures))
        return failures

    monkeypatch.setattr("github_harness_agent.review_solver.evidence", record)
    if scenario in {"revert", "six_hunks"}:
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert all(line in result.patch for line in needed)
        assert rejected not in result.patch
        assert len(gateway.prompts) == 1
        assert result.route_trace[-1].configured_model == "stand-in/free"
        assert probes == [(result.patch, {})]
    else:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert "Private regression repair" in gateway.prompts[1]
        assert len(probes) == (2 if scenario == "required_hunk" else 0)
        assert all(failures for _, failures in probes)
    assert not checkpoints


def test_compose_line_edits_requires_exact_base_and_reviewed_paths():
    from github_harness_agent.review_composition import compose_line_edits, small_line_edits

    base = {SOURCE: "left = 1\nright = 2\n", "tests/test_existing.py": "pass\n"}
    allowed = {SOURCE}
    first = small_line_edits(base, {**base, SOURCE: "left = 3\nright = 2\n"}, allowed)
    second = small_line_edits(base, {**base, SOURCE: "left = 1\nright = 4\n"}, allowed)
    touching = small_line_edits(
        base, {**base, SOURCE: "left = 1\ncenter = 0\nright = 2\n"}, allowed
    )

    assert compose_line_edits(base, first, second) == {
        **base, SOURCE: "left = 3\nright = 4\n"
    }
    assert compose_line_edits(base, first, touching) is None
    assert compose_line_edits({**base, SOURCE: "left = 8\nright = 2\n"}, first, second) is None
    assert small_line_edits(base, {**base, "tests/test_existing.py": "raise RuntimeError\n"}, allowed) == ()


@pytest.mark.parametrize("scenario", ("disjoint", "conflict", "canary"))
def test_solver_combines_only_verified_disjoint_stock_edits(solver_case, monkeypatch, scenario):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    operator = Path(task.pull_checks[0][-1])
    operator.write_text(
        "from fixture_calc import first, second\n\n"
        "def test_first():\n    assert first() == -1\n\n"
        "def test_second():\n    assert second() == -2\n"
    )
    canary = (
        "\ndef test_independent_modes():\n    assert (left(), right()) != (12, 22)\n"
        if scenario == "canary"
        else ""
    )
    sources = {
        SOURCE: BASE_SOURCE + "\ndef left():\n    return 11\n\ndef right():\n    return 21\n",
        "tests/test_existing.py": (
            "from fixture_calc import left, right\n\n"
            "def test_stock_goal():\n    assert left() + right() == 34\n" + canary
        ),
    }
    if scenario == "disjoint":
        prior_source = "# Accepted configuration\n" + sources[SOURCE]
        accepted = DraftCheckpoint(
            "harness2/base",
            task.base_sha,
            patch_for({SOURCE: sources[SOURCE]}, {SOURCE: prior_source}),
            guidance=_GUIDANCE,
        )
        task = replace(task, checkpoints={accepted.key: accepted})
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, source in sources.items():
            data = source.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            completion = super().complete(**kwargs)
            completion.configured_model = f"stand-in/free-{len(self.prompts)}"
            return completion

    left = edit(SOURCE, "    return 11\n", "    return 12\n")
    right = (
        edit(SOURCE, "    return 11\n", "    return 14\n")
        if scenario == "conflict"
        else edit(SOURCE, "    return 21\n", "    return 22\n")
    )
    gateway = Gateway(left, right)
    executor = PlainExecutor(runner, gateway, Limits(model_requests=2, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)
    composed_failures = []

    def record(*args, **kwargs):
        failures = evidence(*args, **kwargs)
        if b"+    return 12\n" in args[4] and b"+    return 22\n" in args[4]:
            composed_failures.append({node for _, node in failures})
        return failures

    monkeypatch.setattr("github_harness_agent.review_solver.evidence", record)
    if scenario == "disjoint":
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert (b"+    return 12\n" in result.patch and b"+    return 22\n" in result.patch)
        assert b"+# Accepted configuration\n" in result.patch
        assert all(check.returncode == 0 for check in result.checks)
        assert [item.configured_model for item in result.route_trace] == [
            "stand-in/free-1", "stand-in/free-2"
        ]
        assert composed_failures == [set()]
    else:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert composed_failures == (
            [{"tests/test_existing.py::test_independent_modes"}] if scenario == "canary" else []
        )
    assert len(gateway.prompts) == 2
    assert not checkpoints


@pytest.mark.parametrize(
    "followup", (
        "repair", "lose_stock", "operator_remains", "partial_stock",
        "malformed", "no_improvement", "retry_limit", "four_cases",
    )
)
def test_private_repair_reconciles_a_stock_gain_without_storing_regressions(solver_case, followup):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    extra_operator = (
        "from fixture_calc import third, ultimate\n\n"
        "def test_third():\n    assert third() == -5\n\n"
        "def test_ultimate():\n    assert ultimate() == -6\n"
        if followup == "four_cases"
        else ""
    )
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, second\n\n"
        "def test_first():\n    assert first() == -1\n\n"
        "def test_second():\n    assert second() == -2\n\n" + extra_operator
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: (
                BASE_SOURCE + "\ndef stock():\n    return -3\n\ndef stock_guard():\n    return 0\n"
                + "\ndef third():\n    return -5\n\ndef ultimate():\n    return -6\n"
            ),
            "tests/test_existing.py": (
                "from fixture_calc import stock, stock_guard\n\n"
                "def test_stock_goal():\n    assert stock() == 3\n\n"
                "def test_stock_guard():\n    assert stock_guard() == 0\n"
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    stock_fix = edit(SOURCE, "    return -3\n", "    return 3\n")
    break_stock_guard = edit(SOURCE, "    return 0\n", "    return 1\n")
    restore_first = edit(SOURCE, "    return 1\n", "    return -1\n")
    restore_second = edit(SOURCE, "    return 2\n", "    return -2\n")
    initial = stock_fix + FIRST + SECOND
    if followup == "four_cases":
        initial += edit(SOURCE, "    return -5\n", "    return 5\n")
        initial += edit(SOURCE, "    return -6\n", "    return 6\n")
    followups = {
        "repair": (restore_first, restore_second),
        "lose_stock": (restore_first + restore_second + break_stock_guard,),
        "operator_remains": (restore_first, edit(SOURCE, "    return 2\n", "    return 4\n")),
        "partial_stock": ("An incomplete edit",),
        "malformed": (
            "An incomplete edit",
            f"REPLACE {SOURCE}::first\ndef first():\n    return -1\n",
            restore_second,
        ),
        "no_improvement": (
            edit(SOURCE, "    return 1\n", "    return 4\n"),
            restore_first,
            restore_second,
        ),
        "retry_limit": ("An incomplete edit",) * 4,
        "four_cases": (
            restore_first,
            restore_second,
            edit(SOURCE, "    return 5\n", "    return -5\n"),
            edit(SOURCE, "    return 6\n", "    return -6\n"),
        ),
    }[followup]
    if followup == "partial_stock":
        initial += break_stock_guard
    after_provider = []

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            after_provider.append(kwargs.get("after_provider"))
            result = super().complete(**kwargs)
            result.route = f"provider-{len(self.prompts)}"
            return result

    gateway = Gateway(initial, *followups)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=len(followups) + 1, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    if followup in {"repair", "malformed", "no_improvement", "four_cases"}:
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert b"+    return 3\n" in result.patch
        assert b"+    return 1\n" not in result.patch
        assert b"+    return 2\n" not in result.patch
        assert b"+    return 4\n" not in result.patch
        assert b"+    return 5\n" not in result.patch
        assert b"+    return 6\n" not in result.patch
        assert all(check.returncode == 0 for check in result.checks)
    else:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")

    assert not checkpoints
    assert "test_stock_goal" in gateway.prompts[0].splitlines()[0]
    if followup in {"partial_stock", "retry_limit"}:
        assert "test_stock_goal" in gateway.prompts[-1].splitlines()[0]
        assert "Private regression repair" not in gateway.prompts[-1]
        if followup == "retry_limit":
            assert sum("Private regression repair" in prompt for prompt in gateway.prompts) == 3
    else:
        cases = {
            "malformed": ("test_first", "test_first", "test_second"),
            "no_improvement": ("test_first", "test_first", "test_second"),
            "four_cases": ("test_first", "test_second", "test_third", "test_ultimate"),
        }.get(followup, ("test_first", "test_second"))
        for prompt, case in zip(gateway.prompts[1:], cases, strict=False):
            assert case in prompt.splitlines()[0]
            assert "Private regression repair" in prompt
            assert "test_stock_goal" in prompt
            assert "assert stock() == 3" in prompt
    if followup == "malformed":
        assert after_provider[2] == "provider-2"
        assert "No complete edit blocks" in gateway.prompts[2]
        assert gateway.prompts[2].endswith("Respond with complete named Python definitions only.")
    if followup == "no_improvement":
        assert after_provider[2] == "provider-2"
        assert "The selected case still fails" in gateway.prompts[2]
    assert len(gateway.prompts) == len(followups) + 1


@pytest.mark.parametrize(
    "scenario", ("mixed", "static_only", "external", "infra", "unrecognized")
)
def test_private_static_repair_preserves_stock_and_rechecks_every_gate(
    solver_case, monkeypatch, scenario
):
    task, _, runner, checkpoints = solver_case
    mypy = ("mypy", ".")
    task = replace(task, checks=(*task.checks, mypy))
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, second, stock\n\n"
        + (
            "def test_first():\n    assert first() == -1\n\n"
            "def test_second():\n    assert second() == -2\n\n"
            "def test_third():\n    assert first() + second() == -3\n"
            if scenario == "static_only"
            else "def test_first():\n    assert isinstance(stock(), int)\n\n"
            "def test_second():\n    assert stock() < 10\n\n"
            "def test_third():\n    assert stock() % 2 == 1 and first() == -1\n"
        )
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\ndef stock() -> int:\n    return -3\n",
            "tests/test_existing.py": (
                "from fixture_calc import stock\n\n"
                "def test_stock_goal():\n    assert str(stock()) == '3'\n"
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    original_command = FixtureWorkspace.command
    mypy_runs = []

    def command(workspace, argv, *args, **kwargs):
        if tuple(argv) != mypy:
            return original_command(workspace, argv, *args, **kwargs)
        source = (workspace.root / SOURCE).read_text()
        mypy_runs.append(source)
        if 'return "3"' not in source:
            return SimpleNamespace(
                returncode=0,
                stdout=b"Success: no issues found in 1 source file\n",
                stderr=b"",
            )
        diagnostic_path = "../outside.py" if scenario == "external" else SOURCE
        diagnostic = (
            "mypy: failed to load project configuration\n"
            if scenario == "unrecognized"
            else f'{diagnostic_path}:8: error: Incompatible return value type (got "str", expected "int")  [return-value]\n'
            "Found 1 error in 1 file (checked 1 source file)\n"
        )
        return SimpleNamespace(
            returncode=2 if scenario == "infra" else 1,
            stdout=diagnostic.encode(),
            stderr=b"",
        )

    monkeypatch.setattr(FixtureWorkspace, "command", command)
    initial = edit(SOURCE, "    return -3\n", '    return "3"\n')
    if scenario == "mixed":
        initial += FIRST
    good = scenario in {"mixed", "static_only"}
    attempts = (
        (
            "An incomplete edit",
            f"REPLACE {SOURCE}::stock\ndef stock() -> int:\n    return -3\n",
            f"REPLACE {SOURCE}::stock\ndef stock() -> int:\n    return 3\n",
            *(
                (edit(SOURCE, "    return 1\n", "    return -1\n"),)
                if scenario == "mixed"
                else ()
            ),
            edit(
                NEW_TEST,
                "",
                "from fixture_calc import stock\n\ndef test_stock_value():\n    assert stock() == 3\n",
            ),
        )
        if good
        else ("An incomplete edit",)
    )
    gateway = FreeGateway(initial, *attempts)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=len(attempts) + 1, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    if good:
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert b"+    return 3\n" in result.patch and NEW_TEST.encode() in result.patch
        assert len(checkpoints) == 1
        assert b"+    return 3\n" in checkpoints[0].patch
        assert b"+    return 1\n" not in checkpoints[0].patch
        for prompt in gateway.prompts[1:4]:
            assert "Private regression repair" in prompt
            assert "check 3 (exit 1)" in prompt.splitlines()[0]
            assert "Incompatible return value type" in prompt
            assert '    return "3"' in prompt
        assert "unchanged stock-passing draft" in gateway.prompts[3]
        if scenario == "mixed":
            assert "test_third" in gateway.prompts[4].splitlines()[0]
            assert "return 3" in gateway.prompts[4]
        assert len(mypy_runs) >= 5
    else:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert not checkpoints
        assert "Private regression repair" not in gateway.prompts[1]


def test_private_regression_repair_is_discarded_if_a_concurrent_draft_advances(
    solver_case,
):
    task, archive, runner, checkpoints = solver_case
    fixture = Path(task.pull_checks[0][-1])
    fixture.write_text(
        fixture.read_text() + "\ndef test_guard():\n"
        "    import fixture_calc\n"
        "    assert not getattr(fixture_calc, 'regression_flag', False)\n"
    )
    repair_started = Event()
    second_saved = Event()
    first_started = Event()
    flag = edit(SOURCE, "def first():\n", "regression_flag = True\n\ndef first():\n")

    def save(checkpoint):
        checkpoints.append(checkpoint)
        if b"+    return 2\n" in checkpoint.patch:
            second_saved.set()

    class Gateway:
        def complete(self, *, role, messages, **request):
            goal = messages[-1]["content"].splitlines()[0]
            if "test_guard" in goal:
                repair_started.set()
                assert second_saved.wait(30)
                proposal = (
                    edit(SOURCE, "regression_flag = True\n", "stale_repair = True\n")
                    + SECOND
                )
            elif "test_second" in goal:
                assert repair_started.wait(30)
                proposal = SECOND
            elif "test_first" in goal:
                proposal = FIRST if first_started.is_set() else FIRST + flag
                first_started.set()
            else:
                proposal = edit(NEW_TEST, "", TEST)
            return SimpleNamespace(
                role=role,
                route="stand-in-free",
                model="fixture-coder",
                configured_model="stand-in/free",
                content=proposal,
                usage={},
            )

    task = replace(task, save_checkpoint=save)
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=5, concurrent_probes=2)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert repair_started.is_set()
    assert b"+    return 2\n" in checkpoints[0].patch
    assert b"+    return 1\n" not in checkpoints[0].patch
    assert all(b"stale_repair" not in draft.patch for draft in checkpoints)
    assert b"stale_repair" not in result.patch


@pytest.mark.parametrize("branch_between", (False, True))
def test_solver_checkpoints_later_straight_line_assertion_only(
    solver_case, branch_between
):
    task, archive, runner, checkpoints = solver_case
    fixture = Path(task.pull_checks[0][-1])
    fixture.write_text(
        "from fixture_calc import first, second\n\n"
        "def test_results():\n"
        "    assert first() == 1\n"
        + ("    if first() != 1:\n        return\n" if branch_between else "")
        + "    assert second() == 2\n"
    )
    operator = (*task.pull_checks[0][:-1], "--tb=line", str(fixture))
    task = replace(task, pull_checks=(operator,))
    gateway = FreeGateway(FIRST)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=1, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    with pytest.raises(
        AttemptRejected, match="outstanding checks still fail"
    ) as rejection:
        solve(executor, task, archive, budget, "fixture-version")

    if branch_between:
        assert not isinstance(rejection.value, ReviewRoundExhausted)
        assert not checkpoints
    else:
        assert isinstance(rejection.value, ReviewRoundExhausted)
        assert len(rejection.value.route_trace) == 1
        assert len(checkpoints) == 1
        assert b"+    return 1\n" in checkpoints[0].patch
        assert b"+    return 2\n" not in checkpoints[0].patch


@pytest.mark.parametrize("native_failure", [None, "exhausted", "out_of_scope"])
def test_solver_tries_one_native_edit_then_keeps_text_fallback(
    solver_case, native_failure
):
    from github_harness_agent.providers import ProviderExhausted

    task, archive, runner, _ = solver_case
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first\n\ndef test_one():\n    assert first() == 1\n"
    )
    task = replace(task, require_new_tests=False)
    calls = []
    replacement = "def first():\n    return 1\n"

    class Gateway(FreeGateway):
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
            calls.append(bool(tools))
            if not tools:
                return super().complete(
                    role=role,
                    messages=messages,
                    tools=tools,
                    timeout=timeout,
                    after_provider=after_provider,
                    max_output_tokens=max_output_tokens,
                )
            assert max_output_tokens == 2048
            assert len(tools) == 1
            function = tools[0]["function"]["name"]
            assert function == "replace_python_definition"
            if native_failure == "exhausted":
                raise ProviderExhausted(None, {"fixture": "tool unavailable"})
            payload = {
                "path": "outside.py" if native_failure else SOURCE,
                "symbol": "first",
                "definition": replacement,
            }
            return SimpleNamespace(
                content=None,
                role=role,
                route="native-free",
                model="fixture-native",
                configured_model="stand-in/native",
                usage={},
                tool_calls=(
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": function,
                            "arguments": json.dumps(payload),
                        },
                    },
                ),
            )

    gateway = Gateway(
        "prose",
        "still prose",
        "invalid named reply",
        f"REPLACE {SOURCE}::first\n{replacement}",
    )
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=5, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive, budget, "fixture-version")

    assert b"+    return 1\n" in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    assert calls == [False, False, False, True] + ([False] if native_failure else [])
    assert result.route_trace[-1].route == (
        "stand-in-free" if native_failure else "native-free"
    )


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


def test_solver_retains_a_reached_and_passing_compound_prefix_after_stagnation(
    solver_case,
):
    task, archive, runner, checkpoints = solver_case
    (check,) = task.pull_checks
    Path(check[-1]).write_text(
        "from fixture_calc import first, second\n\n"
        "def test_compound():\n"
        "    assert first() == 1 and second() == 2, 'compound predicate'\n"
    )
    task = replace(task, pull_checks=((*check[:-1], "--tb=line", check[-1]),))
    continuation = edit(SOURCE, "    return -2\n", "    return 0\n")
    finish = edit(SOURCE, "    return 0\n", "    return 2\n")
    gateway = FreeGateway(FIRST, continuation, finish, edit(NEW_TEST, "", TEST))
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=4, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert b"+    return 1\n" in checkpoints[0].patch
    assert b"+    return 0\n" in checkpoints[0].patch
    assert "first() == 1: passed" in checkpoints[0].progress[0]
    assert "second() == 2: failed" in gateway.prompts[2]
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
        FIRST,
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
    assert gateway.prompts[3].splitlines()[0].endswith("::test_first")
    assert "Private regression repair" in gateway.prompts[3]
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


@pytest.mark.parametrize("late_failure", ("stock", "focused", "setup-only"))
def test_solver_repairs_an_unpublished_regression_test_in_place(
    solver_case, late_failure
):
    task, archive, runner, checkpoints = solver_case
    bad_return = "(0, 0)" if late_failure == "focused" else "(1, 2)  # rejected-style"
    if late_failure == "stock":
        lint = (
            "from pathlib import Path\n"
            "p = Path('tests/test_review_regressions.py')\n"
            "if p.exists() and 'rejected-style' in p.read_text():\n"
            "    raise SystemExit('tests/test_review_regressions.py:4: invalid style')\n"
        )
        task = replace(task, checks=(*task.checks, (sys.executable, "-c", lint)))
    if late_failure == "setup-only":
        bad_test = TEST.split("\n\n", 1)[0] + "\n"
        repair = edit(NEW_TEST, bad_test, TEST)
        expected_diagnostic = "Add at least one runnable pytest test"
    else:
        bad_test = TEST.replace("(1, 2)", bad_return)
        repair = edit(
            NEW_TEST,
            f"    assert (first(), second()) == {bad_return}\n",
            "    assert (first(), second()) == (1, 2)\n",
        )
        expected_diagnostic = (
            "invalid style" if late_failure == "stock" else "test_results"
        )
    gateway = FreeGateway(FIRST, SECOND, edit(NEW_TEST, "", bad_test), repair)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=4, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    repair_prompt = gateway.prompts[3]
    assert bad_test in repair_prompt
    assert expected_diagnostic in repair_prompt
    assert all(NEW_TEST.encode() not in checkpoint.patch for checkpoint in checkpoints)
    assert NEW_TEST.encode() in result.patch
    assert all(check.returncode == 0 for check in result.checks)

    request = gateway.prompts[2].replace(
        "\nCurrent checkout:\n", "\nCurrent checkout:\n" + ("unused context\n" * 2000)
    )
    compact = compact_focused_review(
        [
            {"role": "system", "content": "Review guidance"},
            {"role": "user", "content": request},
        ],
        max_bytes=12 * 1024,
    )[-1]["content"]
    assert "Correct the first result." in compact
    assert "Correct the second result." in compact
    assert "test_first" in compact
    assert "test_second" in compact
    assert "from fixture_calc import first, second" in compact
    assert "def first()" in compact
    assert "fresh offline" in compact


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


@pytest.mark.parametrize("named_failure", ["malformed", "nonimproving"])
def test_localized_named_failure_tries_numbered_ranges_as_an_independent_remedy(
    solver_case, monkeypatch, named_failure
):
    from github_harness_agent import review_solver

    task, _, runner, _ = solver_case
    task = replace(task, require_new_tests=False, excluded_models=("blocked/model",))
    archive = BytesIO()
    stock_method = "def stock_result():\n    return -3\n"
    stock_test = (
        "from fixture_calc import stock_result\n\n"
        "def test_stock_result():\n    assert stock_result() == 3\n"
    )
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\n" + stock_method,
            "tests/test_existing.py": stock_test,
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    def localize(
        executor,
        task,
        archive,
        budget,
        command,
        node,
        original,
        current,
        fixture,
        make_patch,
    ):
        line = current[SOURCE].splitlines().index("    return -3") + 1
        return (
            "Auxiliary counterfactual (quoted data, partial diagnostic only). "
            f"{SOURCE}::stock_result, draft line {line}: causal evidence."
        )

    monkeypatch.setattr(review_solver, "stock_guard_diagnostic", localize)
    calls = []
    edits = iter(
        (
            FIRST + SECOND,
            edit(SOURCE, "    return -3\n", "    return -4\n"),
            edit(SOURCE, "    return -3\n", "    return -5\n"),
        )
    )

    class Gateway:
        def complete(self, *, role, messages, max_output_tokens, **request):
            prompt = messages[-1]["content"]
            system = messages[0]["content"]
            calls.append((role, system, prompt, request, max_output_tokens))
            if role == "planning":
                content = json.dumps(
                    {
                        "strategy": "Trace the public inputs.",
                        "symbols": [f"{SOURCE}::stock_result"],
                    }
                )
            elif "single line-range JSON object" in prompt:
                snapshot = re.search(r"\"snapshot\":\"([a-f0-9]{64})\"", prompt)[1]
                content = json.dumps(
                    {
                        "snapshot": snapshot,
                        "edits": [{"start": 2, "end": 2, "lines": ["    return 3"]}],
                    }
                )
            elif prompt.endswith(
                "Respond with complete named Python definitions only."
            ):
                content = (
                    "bad named response"
                    if named_failure == "malformed"
                    else f"REPLACE {SOURCE}::stock_result\ndef stock_result():\n    return -8"
                )
            elif "Trace the public inputs." in prompt:
                content = "bad search response"
            else:
                content = next(edits)
            return SimpleNamespace(
                role=role,
                route="earlier-format-route",
                model="fixture-coder",
                configured_model="stand-in/free",
                content=content,
                usage={},
            )

    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=7, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive.getvalue(), budget, "fixture-version")

    assert [item[0] for item in calls] == ["coding"] * 3 + ["planning"] + ["coding"] * 3
    assert calls[4][2].endswith("Respond with exact SEARCH/REPLACE blocks only.")
    assert calls[5][2].endswith("Respond with complete named Python definitions only.")
    _, system, prompt, request, max_tokens = calls[6]
    assert "single line-range JSON object" in prompt
    assert "1 | def stock_result():" in prompt and "2 |     return -3" in prompt
    assert prompt.count(stock_method) == 1
    assert "Complete pinned baseline definition" in prompt
    assert "causal evidence" in prompt and "Trace the public inputs." in prompt
    assert "assert stock_result() == 3" in prompt
    assert "Review notes" in prompt
    assert "SEARCH" not in system and "REPLACE" not in system
    assert "bad named response" not in prompt and "return -8" not in prompt
    assert request.get("after_provider") is None
    assert request.get("exclude") == ("blocked/model",)
    assert max_tokens == 6144
    assert system.endswith(review_solver.CAUSAL_EDIT_GUIDANCE)
    assert all(check.returncode == 0 for check in result.checks)
    assert (
        b"+    return 3\n" in result.patch
        and b"tests/test_existing.py" not in result.patch
    )


@pytest.mark.parametrize("interactive_tests", (False, True))
def test_saved_stock_plateau_tries_scoped_tools_then_verifies_regression(
    solver_case, monkeypatch, interactive_tests
):
    from github_harness_agent import review_solver

    task, _, runner, checkpoints = solver_case
    recovery = []
    task = replace(task, save_solver_recovery=lambda state, durable: recovery.append((state, durable)))
    if interactive_tests:
        task = replace(task, checks=((sys.executable, "-m", "pytest", "-q", "tests"),))
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\ndef stock_result():\n    return -3\n",
            "tests/test_existing.py": (
                "from fixture_calc import stock_result\n\ndef test_stock_result():\n    assert stock_result() == 3\n"
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    monkeypatch.setattr(review_solver, "mock_call_diagnostic", lambda *args: "")
    monkeypatch.setattr(
        review_solver, "stock_guard_diagnostic",
        lambda *args: "Auxiliary counterfactual (quoted data, partial diagnostic only). fixture_calc.py::stock_result, draft line 8: causal evidence.",
    )
    text_edits = iter((
        FIRST + SECOND,
        edit(SOURCE, "    return -3\n", "    return -4\n"),
        edit(SOURCE, "    return -3\n", "    return -5\n"),
    ))
    phases = []
    tool_turns = []
    regression = "from fixture_calc import stock_result\n\ndef test_repair():\n    assert stock_result() == 3\n"

    def call(name, arguments):
        return {"id": name + arguments.get("path", ""), "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    class Gateway:
        def complete(self, *, role, messages, tools, timeout, **kwargs):
            prompt = messages[-1]["content"]
            calls = ()
            if tools:
                tool_turns.append(timeout)
                phases.append("interactive")
                outputs = [message["content"] for message in messages if message["role"] == "tool"]
                if not outputs:
                    task_prompt = "\n".join(str(m.get("content") or "") for m in messages)
                    assert "Localized method in the current draft" in task_prompt
                    assert "def stock_result():\n    return -3" in task_prompt
                    assert {tool["function"]["name"] for tool in tools} >= {"run", "edit_file"}, phases
                    calls = (
                        call("edit_file", {"path": "outside.py", "new_text": "bad = True\n"}),
                        call("run", {"argv": ["cat", SOURCE]}),
                    )
                elif len(tool_turns) == 2:
                    assert "outside the scoped review" in outputs[0]
                    assert "return 1" in outputs[1] and "return 2" in outputs[1]
                    assert "return -3" in outputs[1]
                    calls = (call("edit_file", {"path": SOURCE, "old_text": "    return -3\n", "new_text": "    return 3\n"}),)
                    if interactive_tests:
                        calls += (call("edit_file", {"path": NEW_TEST, "new_text": regression}),)
                elif len(tool_turns) == 3:
                    calls = (call("run", {"argv": list(task.checks[0])}),)
                else:
                    assert ("2 passed" if interactive_tests else "1 passed") in outputs[-1]
                content = "done" if not calls else None
            elif role == "planning":
                phases.append("planning")
                content = json.dumps({"strategy": "Trace the public inputs.", "symbols": [f"{SOURCE}::stock_result"]})
            elif prompt.startswith("Selected failure: new regression tests"):
                phases.append("author")
                assert len(checkpoints) == 1 and b"+    return 3\n" not in checkpoints[0].patch
                content = edit(NEW_TEST, "", regression)
            elif "single line-range JSON object" in prompt:
                phases.append("numbered")
                content = "invalid line range"
            elif prompt.endswith("Respond with complete named Python definitions only."):
                phases.append("named")
                content = "invalid named edit"
            elif "Trace the public inputs." in prompt:
                phases.append("guided search")
                content = "invalid search edit"
            else:
                phases.append("text")
                content = next(text_edits)
            return SimpleNamespace(
                role=role, route="fixture", model="fixture", configured_model="fixture", content=content,
                tool_calls=calls, usage={},
            )

    verifier_root = runner.path / "verification"
    verifier_root.mkdir()
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=14, concurrent_probes=1),
        verification_runner=FixtureRunner(verifier_root),
    )
    result = solve(executor, task, archive.getvalue(), _Budget(executor.limits, lambda: False, task), "fixture")

    expected = ["text"] * 3 + ["planning", "guided search", "named", "numbered"] + ["interactive"] * 4
    if not interactive_tests:
        expected.append("author")
    assert phases == expected
    assert len(checkpoints) == 1 and b"+    return 3\n" not in checkpoints[0].patch
    assert any(durable for _, durable in recovery)
    assert all(0 < timeout <= 120 for timeout in tool_turns)
    assert b"+    return 3\n" in result.patch and NEW_TEST.encode() in result.patch
    assert b"outside.py" not in result.patch
    assert all(check.returncode == 0 for check in result.checks)


def test_scoped_rollout_rejects_unreviewed_files_written_through_run(solver_case):
    task, archive, runner, _ = solver_case
    task = replace(task, save_checkpoint=None, review_recovery=None, require_new_tests=False)

    class Gateway:
        def complete(self, *, role, messages, **request):
            finished = any(message["role"] == "tool" for message in messages)
            commands = () if finished else ({
                "id": "run",
                "type": "function",
                "function": {
                    "name": "run",
                    "arguments": json.dumps({
                        "argv": [sys.executable, "-c", "from pathlib import Path; Path('outside.py').write_text('pass\\n')"]
                    }),
                },
            },)
            return SimpleNamespace(
                role=role, route="fixture", model="fixture", configured_model="fixture",
                content="done" if finished else None, tool_calls=commands, usage={},
            )

    executor = PlainExecutor(runner, Gateway(), Limits(model_requests=2))
    budget = _Budget(executor.limits, lambda: False, task)
    scope = {SOURCE: BASE_SOURCE}
    with pytest.raises(AttemptRejected, match="Interactive edits exceeded the scoped review paths"):
        executor._rollout(task, archive, budget, "Test scoped tool calls.", "coding", clarification=False, scoped_sources=scope)
    assert scope == {SOURCE: BASE_SOURCE}
