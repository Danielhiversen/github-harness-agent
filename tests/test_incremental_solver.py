import json
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
        lambda *args: {(1, "check 2 (exit 1)"): f"{SOURCE}:1:1: I001 Imports unsorted"},
    )

    with pytest.raises(AttemptRejected):
        solve(executor, task, archive, budget, "fixture-version")

    assert len(gateway.prompts) == 3
    assert all(
        prompt.endswith("Respond with exact SEARCH/REPLACE blocks only.")
        for prompt in gateway.prompts
    )


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
                assert timeout <= 25
                assert request["max_output_tokens"] <= 512
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

    with pytest.raises(AttemptRejected, match="outstanding checks still fail"):
        solve(executor, task, archive, budget, "fixture-version")

    if branch_between:
        assert not checkpoints
    else:
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
