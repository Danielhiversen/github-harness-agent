import json
import sys
import tarfile
from pathlib import Path
from dataclasses import replace
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from test_executor import FixtureRunner
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

    def verify(executor, task, archive, budget, patch, commands, *, pytest_runs=None):
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
    monkeypatch.setattr("github_harness_agent.review_solver.mock_call_diagnostic", lambda *args: "Observed differing calls.")
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
    assert b"+enabled = True\n" in result.patch
    assert not any(b"+enabled = False" in checkpoint.patch for checkpoint in checkpoints)
    assert all(check.returncode == 0 for check in result.checks)


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
    "scenario", ("compatible", "identical", "canary", "unrelated", "low_budget")
)
def test_guarded_stock_reconciliation_checks_every_suite(solver_case, monkeypatch, scenario):
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
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\ndef stock(*, legacy: bool = False):\n    shared = 1\n    return shared - 3\n",
            "tests/test_existing.py": (
                "from fixture_calc import stock\n\n"
                "def test_stock_goal():\n    assert stock() == 4\n" + canary
            ),
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    proposal = edit(SOURCE, "    return shared - 3\n", "    return shared + 3\n")
    if scenario == "unrelated":
        proposal += SECOND
    gateway = FreeGateway(proposal, "An incomplete edit")
    executor = PlainExecutor(runner, gateway, Limits(model_requests=2, concurrent_probes=1))
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
    if scenario == "compatible":
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert b"shared - 3 if legacy else shared + 3" in result.patch
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


@pytest.mark.parametrize("followup", ("repair", "lose_stock", "operator_remains", "partial_stock"))
def test_private_repair_reconciles_a_stock_gain_without_storing_regressions(solver_case, followup):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, second\n\n"
        "def test_first():\n    assert first() == -1\n\n"
        "def test_second():\n    assert second() == -2\n"
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: BASE_SOURCE + "\ndef stock():\n    return -3\n\ndef stock_guard():\n    return 0\n",
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
    followups = {
        "repair": (restore_first, restore_second),
        "lose_stock": (restore_first + restore_second + break_stock_guard,),
        "operator_remains": (restore_first, edit(SOURCE, "    return 2\n", "    return 4\n")),
        "partial_stock": ("An incomplete edit",),
    }[followup]
    if followup == "partial_stock":
        initial += break_stock_guard
    gateway = FreeGateway(initial, *followups)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=len(followups) + 1, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    if followup == "repair":
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert b"+    return 3\n" in result.patch
        assert b"+    return 1\n" not in result.patch
        assert b"+    return 2\n" not in result.patch
        assert all(check.returncode == 0 for check in result.checks)
    else:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")

    assert not checkpoints
    assert "test_stock_goal" in gateway.prompts[0].splitlines()[0]
    if followup == "partial_stock":
        assert "test_stock_goal" in gateway.prompts[1].splitlines()[0]
        assert "Private regression repair" not in gateway.prompts[1]
    else:
        for prompt, case in zip(gateway.prompts[1:], ("test_first", "test_second")):
            assert case in prompt.splitlines()[0]
            assert "Private regression repair" in prompt
            assert "test_stock_goal" in prompt
            assert "assert stock() == 3" in prompt
    assert len(gateway.prompts) == len(followups) + 1


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
