import json
import re
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from dataclasses import replace
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from test_executor import FixtureRunner, FixtureWorkspace
from test_providers import (
    KILO, LOCAL_MODEL, OLLAMA, OPENROUTER, OPENROUTER_SIBLING,
    deployment, local_inventory, mock_async_local, pinned_local, routed,
)

from github_harness_agent import review_recovery
from github_harness_agent.checkpoint import DraftCheckpoint, SolverRecovery
from github_harness_agent.providers import ProviderExhausted
from github_harness_agent.executor import (
    AttemptRejected,
    Limits,
    PlainExecutor,
    ReviewRoundExhausted,
    TaskSnapshot,
    _Budget,
)
from github_harness_agent.harness2_adapter import verify_review_repair
from github_harness_agent.review_assertions import (
    assertion_progress,
    operator_assertions,
)
from github_harness_agent.review_recovery import ReviewRecovery, plan_recovery
from github_harness_agent.task_context import compact_focused_review, wire_size
from github_harness_agent.review_solver import (
    LINE_RANGE_ENDING,
    _GUIDANCE,
    _generic_verifier_failure,
    _rejected_attempt,
    _line_failure_details,
    apply_blocks,
    evidence,
    patch_for,
    solve,
)
from github_harness_agent.task_context import excerpt


from github_harness_agent.worker import CommandOutputExceeded, ExecutionError

SOURCE = "fixture_calc.py"
NEW_TEST = "tests/test_review_regressions.py"
BASE_SOURCE = "def first():\n    return -1\n\ndef second():\n    return -2\n"


@pytest.mark.parametrize("report,expected_category,expected_message,expected_location", (
    (
        "===== ERRORS =====\n"
        "_____ ERROR collecting tests/test_example.py _____\n"
        "tests/test_example.py:7: in <module>\n"
        "E   ModuleNotFoundError: No module named 'missing_backend'\n"
        "----- Captured stdout -----\n"
        "E   TypeError: printed-not-reported\n",
        "ModuleNotFoundError", "missing_backend", "tests/test_example.py:7",
    ),
    (
        "===== ERRORS =====\n"
        "_____ ERROR at setup of test_example _____\n"
        "file /work/tests/test_example.py, line 11\n"
        "E   fixture 'unknown_fixture' not found\n",
        "fixture not found", "unknown_fixture", "/work/tests/test_example.py:11",
    ),
    (
        "===== FAILURES =====\n"
        "tests/test_example.py:14: Failed: DID NOT RAISE <class 'ValueError'>\n",
        "DID NOT RAISE", "DID NOT RAISE", "tests/test_example.py:14",
    ),
    (
        "===== short test summary info =====\n"
        "ERROR tests/test_example.py - SyntaxError: unexpected token\n",
        "SyntaxError", "unexpected token", "tests/test_example.py",
    ),
))
def test_generic_pytest_failure_selects_structured_details_before_excerpting(
    report, expected_category, expected_message, expected_location,
):
    output = (
        "early noise\n" * 500 + report
        + "===== warnings summary =====\n" + "late noise\n" * 500
    )
    assert expected_message not in excerpt(output, 3000)
    detail, kind = _generic_verifier_failure(("uv", "run", "pytest"), 2, output)
    assert kind == expected_category
    assert expected_message in detail and expected_location in detail
    assert not any(marker in detail for marker in ("printed-not-reported", "early noise", "late noise"))
    command = ("uv", "run", "pytest")
    executor = SimpleNamespace(
        verification_runner=SimpleNamespace(workspace=lambda *args: nullcontext(object())),
        _stage_baseline=lambda *args: None,
        _command=lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout=output.encode(), stderr=b""),
    )
    budget = SimpleNamespace(
        remaining=None, cancelled=None, claim=lambda *args: None, limits=SimpleNamespace(commands=1),
    )
    labels = set()
    assert evidence(executor, None, None, budget, b"", (command,), test_check_labels=labels) == {
        (0, "check 1 (exit 2)"): detail,
    }
    assert labels == {f"pytest/interrupted: {expected_category}"}


@pytest.mark.parametrize("command,output", (
    (("pytest",), "ordinary output\nE   TypeError: printed\nDID NOT RAISE\n"),
    (("pytest",), "===== FAILURES =====\n----- Captured log call -----\nE   TypeError: printed\n"),
    (("ruff", "check"), "===== ERRORS =====\n_____ ERROR collecting tests/test_example.py _____\n"),
))
def test_generic_verifier_failure_leaves_unstructured_and_non_pytest_output_alone(command, output):
    assert _generic_verifier_failure(command, 1, output) == (output, "")


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


@pytest.mark.parametrize("scenario", (
    "ordinary", "nested_ordinary", "cross_ordinary", "cross_numbered",
    "malformed", "nested", "no_change", "rejected", "outside",
))
def test_operator_limit_uses_traversal_windows_and_verified_retry(
    solver_case, scenario,
):
    task, _, runner, _ = solver_case
    nested = scenario in {"nested", "nested_ordinary", "cross_ordinary", "cross_numbered"}
    cross_key = scenario.startswith("cross_")
    ordinary_only = scenario in {"ordinary", "nested_ordinary", "cross_ordinary"}
    excluded = (("excluded", "configured-excluded"),)
    task = replace(task, require_new_tests=False, excluded_models=excluded)
    acceptance = Path(task.pull_checks[0][-1])
    with acceptance.open("a") as stream:
        stream.write(
            "\nfrom fixture_calc import Explorer\n\n"
            "def test_limit_applies_to_traversal():\n"
            f"    assert Explorer().walk(limit=1) == {[0, 0, 1, 1] if nested else [0, 1]}\n"
        )
    filler = "# Unrelated details elsewhere in the reviewed production module.\n" * 420
    prior = (
        BASE_SOURCE + filler + "\nclass Explorer:\n"
        "    def walk(self, *, limit):\n"
        "        result = [0]\n"
        "        for level in range(limit):\n"
        "            result.append(level + 1)\n"
        "        return result\n"
    )
    current = (
        BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
        + filler
        + "\nclass State:\n"
        "    def __init__(self, depth):\n"
        "        self.depth = depth\n\n"
        "class Explorer:\n"
        "    def walk(self, *, limit):\n"
        "        if limit < 0:\n"
        "            raise ValueError('nonnegative limit required')\n"
        "        pending = [State(0)]\n"
        "        result = []\n"
        "        while pending:\n"
        "            state = pending.pop()\n"
        "            result.append(state.depth)\n"
        "            if state.depth < 2:\n"
        "                child = self._descend(state)\n"
        "                pending.append(child)\n"
        "        return result\n\n"
        "    def _descend(self, parent):\n"
        "        return State(parent.depth + 1)\n"
    )
    expected_sites = (
        "        pending = [State(0)]",
        "            state = pending.pop()",
        "                child = self._descend(state)",
        "                pending.append(child)",
        "    def _descend(self, parent):",
        "        return State(parent.depth + 1)",
    )
    if nested:
        prior = prior.replace("result = [0]", "result = [0, 0]").replace(
            "result.append(level + 1)", "result.extend([level + 1] * 2)"
        )
        before, _, tail = current.partition("        pending = [State(0)]\n")
        _, _, after = tail.partition("        return result\n")
        current = (
            before.replace(
                "\nclass State:", "\nfrom collections import deque\n\nclass State:"
            )
            + "        scheduler = ['b', 'a']\n"
            "        work = {lane: deque([State(0)]) for lane in scheduler}\n"
            "        result = []\n"
            "        while scheduler:\n"
            "            lane = scheduler.pop(0)\n"
            "            local = work[lane]\n"
            "            state = local.popleft()\n"
            "            result.append(state.depth)\n"
            "            if state.depth < 2:\n"
            "                raw_parts = self._descend(state)\n"
            "                parts = [part for part in raw_parts]\n"
            "                work.setdefault(lane, deque()).extendleft(parts)\n"
            "            if local:\n"
            "                scheduler.append(lane)\n"
            "        return result\n"
            + after.replace(
                "return State(parent.depth + 1)", "return [State(parent.depth + 1)]"
            )
        )
        expected_sites = (
            "        scheduler = ['b', 'a']",
            "            lane = scheduler.pop(0)",
            "                scheduler.append(lane)",
            "            state = local.popleft()",
            "                raw_parts = self._descend(state)",
            "                parts = [part for part in raw_parts]",
            "                work.setdefault(lane, deque()).extendleft(parts)",
            "    def _descend(self, parent):",
            "        return [State(parent.depth + 1)]",
            "class State:",
            "        self.depth = depth",
        )
    if cross_key:
        current = current.replace("local = work[lane]", "local = work.get(lane, deque())")
        enqueue = (
            "                    work.setdefault(key, deque()).extendleft(reversed(child_list))"
            + "  # "
            + "documented ordering " * 9
        )
        current = current.replace(
            "        self.depth = depth\n",
            "        self.depth = depth\n\n"
            "    def code(self):\n        return self.depth\n\n"
            "    def expand(self):\n        return [State(self.depth + 1)]\n",
        ).replace(
            "            if state.depth < 2:\n"
            "                raw_parts = self._descend(state)\n"
            "                parts = [part for part in raw_parts]\n"
            "                work.setdefault(lane, deque()).extendleft(parts)\n",
            "            classification = state.code()\n"
            "            if state.depth < 2:\n"
            "                parts = state.expand()\n"
            "                groups = {}\n"
            "                for child in parts:\n"
            "                    groups.setdefault(lane, []).append(child)\n"
            "                for key, child_list in groups.items():\n"
            "                    # Pass all grouped items to the appropriate worklist.\n"
            + enqueue
            + "\n",
        )
        expected_sites = (
            *expected_sites[:4],
            "            if state.depth < 2:\n".rstrip("\n"),
            "                parts = state.expand()",
            "                for child in parts:",
            enqueue,
        )
    checkpoint = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: prior}, {SOURCE: current}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={checkpoint.key: checkpoint})
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: prior,
            "tests/test_existing.py": "def test_stock():\n    assert True\n",
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))

    class Gateway(FreeGateway):
        calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            messages = kwargs["messages"]
            compact = compact_focused_review(messages, max_bytes=8192) or messages
            prompt = compact[-1]["content"]
            assert kwargs["exclude"] == excluded
            assert (
                prompt.startswith("Selected failure:")
                and "test_limit_applies_to_traversal" in prompt.splitlines()[0]
            )
            assert wire_size(compact) <= 8192
            if self.calls == 2 and not ordinary_only:
                assert kwargs["after_provider"] == "stand-in-free"
                assert kwargs["max_output_tokens"] == 2048
                assert prompt.endswith("Respond with the single line-range JSON object only.")
                assert "Pinned iteration bound" not in prompt and "Current checkout:" not in prompt
                signature = re.search(r'"snapshot":"([a-f0-9]+)"', prompt)[1]
                child = (
                    "                raw_parts = self._descend(state)"
                    if nested else "                child = self._descend(state)"
                )
                if cross_key:
                    child = "                parts = state.expand()"
                row = int(re.search(r"(?m)^(\d+) \| " + re.escape(child) + "$", prompt)[1])
                adjusted = "limit + 1" if scenario == "rejected" else "limit"
                lines = [
                    f"                if state.depth >= {adjusted}:",
                    "                    continue",
                    child,
                ]
                if nested:
                    scheduled = "extendleft(reversed(child_list))" if cross_key else "extendleft(parts)"
                    assert "state = local.popleft()" in prompt and scheduled in prompt
                    lines = [child + " if state.depth < limit else []"]
                if scenario == "outside":
                    row = 2  # Unshown validation row, inside the definition body.
                    assert not re.search(r"(?m)^2 \|", prompt)
                return SimpleNamespace(
                    content=json.dumps({"snapshot": signature, "edits": [
                        {"start": row, "end": row, "lines": lines},
                    ]}),
                    role="coding", route="numbered-free", model="fixture-coder",
                    configured_model="numbered/free", usage={}, tool_calls=(),
                )
            focus = prompt.split("\nFocused source from ", 1)[1].split(
                "\nFailure details:", 1
            )[0]
            for row in expected_sites:
                assert f"{current.splitlines().index(row) + 1} | {row}" in focus
            if nested:
                assert "Candidate sites across worklists" in focus
            if cross_key:
                assert "result.append(state.depth)" not in focus
                assert "classification = state.code()" not in focus
            prior_row = "        for level in range(limit):"
            assert f"{prior.splitlines().index(prior_row) + 1} | {prior_row}" in focus
            assert (
                "keyword limit" in focus
                and "Pinned iteration bound (read-only)" in focus
            )
            assert prompt.endswith("Respond with exact SEARCH/REPLACE blocks only.")
            return super().complete(**kwargs)

    ordinary = edit(
        SOURCE,
        "            if state.depth < 2:\n",
        "            if state.depth < min(2, limit):\n",
    )
    first = (
        edit(SOURCE, "        result = []\n", "        result = list()\n")
        if scenario == "no_change"
        else f"{SOURCE}\n<<<<<<< SEARCH\n        result = []\n"
    )
    reports = 1 if ordinary_only else 3 if scenario in {"rejected", "outside"} else 2
    gateway = Gateway(ordinary) if ordinary_only else Gateway(first, ordinary)
    verification = runner.path / "verification"
    verification.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=reports, concurrent_probes=1),
        verification_runner=FixtureRunner(verification),
    )
    result = solve(
        executor,
        task,
        archive.getvalue(),
        _Budget(executor.limits, lambda: False, task),
        "fixture",
    )
    assert all(check.returncode == 0 for check in result.checks)
    assert gateway.calls == reports
    assert b"result = list()" not in result.patch and b"limit + 1" not in result.patch
    if scenario == "rejected":
        assert any(
            "state.depth >= limit + 1" in (work / SOURCE).read_text()
            for work in executor.verification_runner.workspaces
            if (work / SOURCE).is_file()
        )
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

    def verify(
        executor, task, archive, budget, patch, commands, *,
        pytest_runs=None, static_runs=None, private_proposal=False,
    ):
        assert not private_proposal
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


@pytest.mark.parametrize(
    ("oversized", "committed_setup"),
    ((False, False), (True, False), (False, True), (False, "retry")),
    ids=("quoted", "oversized", "committed-setup", "prior-green-seed"),
)
def test_single_pinned_production_review_fixes_all_checks_before_authoring_regression(
    solver_case, oversized, committed_setup,
):
    task, archive, runner, _ = solver_case
    if oversized:
        acceptance = Path(task.pull_checks[0][-1])
        with acceptance.open("a") as stream:
            stream.write("# withheld-operator-source " + "x" * review_recovery.MAX_OPERATOR_SOURCE + "\n")
    test_path = NEW_TEST
    if committed_setup:
        addition = BytesIO(archive)
        previous = (
            b"import pytest\n"
            b"from fixture_calc import first\n"
            b"def make_config(tmp_path):\n"
            b"    return tmp_path / 'sample.json'\n"
            b"@pytest.fixture\n"
            b"def sample_config(tmp_path):\n"
            b"    return make_config(tmp_path)\n"
            b"def test_previous(sample_config):\n"
            b"    assert first() in {-1, 1}, 'old assertion marker'\n"
        )
        with tarfile.open(fileobj=addition, mode="a") as snapshot:
            entry = tarfile.TarInfo(NEW_TEST)
            entry.size = len(previous)
            snapshot.addfile(entry, BytesIO(previous))
        archive = addition.getvalue()
        test_path = NEW_TEST.removesuffix(".py") + "_2.py"
    body = (
        "Correct the first and second results and cover both values in one scenario.\n\n"
        "Update production code and add a new regression test.\n"
        f"Verify regression against: {task.base_sha}"
    )
    review = plan_recovery([{
        "key": "comment:1", "author": "trusted-reviewer", "thread": "review", "reply_to": 1,
        "path": SOURCE, "line": 2, "body": body,
    }], 0, "")
    assert review is not None and not review.test_only
    task = replace(
        task, inline_review_threads=1, review_recovery=review,
        regression_base=task.base_sha, regression_archive=archive, regression_edit_path=SOURCE,
    )
    proposals = (TEST,)
    if committed_setup == "retry":
        missing = "def test_missing(missing_setup):\n    assert missing_setup\n"
        still_missing = missing.replace("missing_setup", "other_setup")
        unchanged = previous.decode()
        corrected = unchanged.replace("import first\n", "import first, second\n").replace(
            "assert first() in {-1, 1}, 'old assertion marker'",
            "assert (first(), second()) == (1, 2)",
        )
        proposals = (
            missing, edit(test_path, missing, still_missing),
            edit(test_path, still_missing, unchanged), edit(test_path, unchanged, corrected),
        )
    gateway = FreeGateway(FIRST, SECOND, *proposals)
    independent = runner.path / "production-regression-verifier"
    independent.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=2 + len(proposals), concurrent_probes=1),
        verification_runner=FixtureRunner(independent),
    )
    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture")

    assert result is not None
    first, second, proof, *retries = gateway.prompts
    assert "test_first" in first and "test_second" in second
    if oversized:
        assert "withheld-operator-source" not in "\n".join(gateway.prompts)
        assert "Operator test names: \n" in proof
    assert f"Editable checkout paths (use these exact names): {SOURCE}" in first
    assert "Correct the first and second results" in first and "Correct the first and second results" in proof
    assert "passes here and fails on the pinned ancestor" in proof
    assert "-    return -1" in proof and "+    return 1" in proof
    assert "def first():\n    return 1" in proof.partition("Optional tracked pytest")[2]
    if committed_setup:
        assert "def sample_config(tmp_path):" in proof
        assert "def test_previous" not in proof and "old assertion marker" not in proof
    if committed_setup == "retry":
        assert "Prior test seed passed on both revisions" not in retries[0]
        assert "Prior test seed passed on both revisions" in retries[1]
        assert "def test_previous(sample_config):" in retries[1]
        assert "def make_config(tmp_path):" in retries[1]
        assert "The new tests also pass on the ancestor" in retries[2]
        assert b"old assertion marker" not in result.patch
        assert f"+++ b/{NEW_TEST}\n".encode() not in result.patch
    assert b"+    return 1\n" in result.patch and b"+    return 2\n" in result.patch
    assert f"+++ b/{test_path}".encode() in result.patch
    assert any(check.command == task.pull_checks[0] and check.returncode == 0 for check in result.checks)
    assert any(check.command[0] == "verify-regression" and check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize("failover", ("none", "malformed", "no_progress", "native", "raw", "raw-restored"))
def test_mixed_review_uses_numbered_retry_without_operator_source(solver_case, failover):
    task, _, runner, checkpoints = solver_case
    raw_mode = failover in {"raw", "raw-restored"}
    operator = Path(task.pull_checks[0][-1])
    with operator.open("a") as stream:
        stream.write("# withheld-operator-source " + "x" * review_recovery.MAX_OPERATOR_SOURCE + "\n")
    production = (
        "class Reader:\n"
        "    def read(self):\n"
        "        # Reviewed dispatch stays in this method.\n"
        + "        # Keep surrounding setup.\n" * (47 if raw_mode else 12)
        + "        def local_result():\n"
        "            return -1\n"
        "        return local_result()\n\n"
        "    def unrelated(self):\n"
        "        return 999\n\n"
        "def first():\n"
        "    return Reader().read()\n\n"
        "def second():\n"
        "    return 2\n"
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for path, body in ((SOURCE, production), ("tests/test_existing.py", "def test_existing(): pass\n")):
            data = body.encode()
            item = tarfile.TarInfo(path)
            item.size = len(data)
            tar.addfile(item, BytesIO(data))
    archive = archive.getvalue()
    body = (
        "Correct the dispatch result without changing the other reviewed value.\n\n"
        "Update production code and add a new regression test.\n"
        f"Verify regression against: {task.base_sha}"
    )
    review = plan_recovery([{
        "key": "comment:1", "author": "trusted-reviewer", "thread": "review", "reply_to": 1,
        "path": SOURCE, "line": 3, "body": body,
    }], 0, "")
    task = replace(
        task, inline_review_threads=1, review_recovery=review, regression_base=task.base_sha,
        regression_archive=archive, regression_edit_path=SOURCE,
    )
    independent = runner.path / "mixed-numbered-verifier"
    independent.mkdir()
    if failover == "raw-restored":
        from github_harness_agent import review_solver

        node = operator.relative_to(independent / "0", walk_up=True)
        target = (0, f"{node}::test_first")
        task = replace(task, solver_recovery=SolverRecovery(
            review_solver._recovery_digest(task, {}, b""),
            ((review_solver._goal_digest(target), 2, 2),),
        ))

    line_attempts = []
    native_attempts = []
    raw_attempts = []

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            if "Return only the complete Python function" in prompt:
                assert raw_mode and not kwargs["tools"]
                reply = super().complete(**kwargs)
                raw_attempts.append(kwargs.get("exclude", ()))
                reply.route = f"raw-free-{len(raw_attempts)}"
                reply.configured_model = f"raw-model-{len(raw_attempts)}"
                return reply
            if kwargs["tools"]:
                tool, = kwargs["tools"]
                schema = tool["function"]
                assert schema["name"] == "replace_python_line_ranges"
                assert schema["parameters"]["required"] == ["edits"]
                assert set(schema["parameters"]["properties"]) == {"edits"}
                prompt = kwargs["messages"][-1]["content"]
                assert '"snapshot"' not in prompt and "def local_result():" in prompt
                native_attempts.append((kwargs["exclude"], kwargs["after_provider"]))
                if failover in {"malformed", "raw"}:
                    raise ProviderExhausted(None, {"fixture": "no native coding route"})
                assert failover == "native"
                reply = super().complete(**{**kwargs, "tools": (), "max_output_tokens": 3072})
                target = re.search(r"(?m)^(\d+) \| ( +)return -1$", prompt)
                row = int(target[1])
                reply.route = "numbered-native-b"
                reply.configured_model = "native-code-model"
                reply.content = ""
                reply.tool_calls = ({
                    "type": "function",
                    "function": {"name": schema["name"], "arguments": json.dumps({
                        "edits": [{"start": row, "end": row, "lines": [target[2] + "return 1"]}],
                    })},
                },)
                return reply
            if kwargs.get("max_output_tokens") != 2048:
                return super().complete(**kwargs)
            reply = super().complete(**{**kwargs, "max_output_tokens": 3072})
            prompt = self.prompts[-1]
            digest = re.search(r'"snapshot":"([0-9a-f]{64})"', prompt)[1]
            target = re.search(r"(?m)^(\d+) \| ( +)return -1$", prompt)
            row = int(target[1])
            first_route = not line_attempts
            reply.route = "numbered-free-a" if first_route else "numbered-free-b"
            reply.configured_model = "small-code-model" if first_route else "alternate-code-model"
            line_attempts.append((digest, kwargs.get("exclude", ()), kwargs.get("after_provider")))
            if failover == "raw" or (first_route and failover in {"malformed", "native"}):
                reply.content = '{"unexpected":"format","edits":[]}'
            else:
                value = -3 if first_route and failover == "no_progress" else 1
                reply.content = json.dumps({
                    "snapshot": digest,
                    "edits": [{"start": row, "end": row, "lines": [target[2] + f"return {value}"]}],
                })
            return reply

    malformed = edit(SOURCE, "    missing_anchor()\n", "    return 1\n")
    numbered_replies = 1 if failover == "none" else 2
    raw_reply = (
        "def wrong_name():\n    return 1\n",
        "def local_result():\n    return -4\n",
        "```python\ndef local_result():\n    return 1\n```",
    ) if failover == "raw" else ()
    if failover == "raw-restored":
        gateway = Gateway("def local_result():\n    return 1\n", TEST)
        requests = 2
    else:
        gateway = Gateway(malformed, malformed, *(["Numbered response expected."] * numbered_replies), *raw_reply, TEST)
        requests = 3 + numbered_replies + (failover in {"malformed", "native", "raw"}) + len(raw_reply)
    executor = PlainExecutor(
        runner, gateway, Limits(
            model_requests=requests,
            concurrent_probes=1,
        ),
        verification_runner=FixtureRunner(independent),
    )
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive, budget, "fixture")

    if failover == "raw-restored":
        assert len(gateway.prompts) == 2 and raw_attempts == [()]
        assert "Reader.read.local_result" in gateway.prompts[0]
        assert any(check.last_editor == ("raw-free-1", "raw-model-1") for check in checkpoints)
        assert any(check.command[0] == "verify-regression" and check.returncode == 0 for check in result.checks)
        return
    numbered = gateway.prompts[2]
    assert len(line_attempts) == (1 if failover == "native" else numbered_replies)
    excluded = (("numbered-free-a", "small-code-model"),)
    if failover in {"malformed", "native", "raw"}:
        assert native_attempts == [(excluded, "numbered-free-a")]
    else:
        assert native_attempts == []
    if failover in {"malformed", "no_progress", "raw"}:
        assert line_attempts[0][0] == line_attempts[1][0]
        assert line_attempts[1][1] == excluded
        assert line_attempts[1][2] == "numbered-free-a"
    if failover == "native":
        assert any(route.route == "numbered-native-b" for route in budget.routes)
        assert any(check.last_editor == ("numbered-native-b", "native-code-model") for check in checkpoints)
    assert "Correct the dispatch result" in numbered and "assert -1 == 1" in numbered
    assert f"{SOURCE}::Reader.read" in numbered and "def local_result():" in numbered
    assert "def unrelated" not in numbered and "withheld-operator-source" not in numbered
    if failover == "raw":
        raw = gateway.prompts[-2]
        assert raw_attempts == [
            (), (("raw-free-1", "raw-model-1"),),
            (("raw-free-1", "raw-model-1"), ("raw-free-2", "raw-model-2")),
        ]
        assert any(check.last_editor == ("raw-free-3", "raw-model-3") for check in checkpoints)
        assert "Return only the complete Python function" in raw
        assert "Reader.read.local_result" in raw and "local_result()" in raw
        assert "Correct the dispatch result" in raw and "assert -1 == 1" in raw
        assert "Keep surrounding setup" not in raw and "withheld-operator-source" not in raw
        assert "def unrelated" not in raw and '"snapshot"' not in raw
    assert b"+            return 1" in result.patch and b"+++ b/tests/test_review_regressions.py" in result.patch
    assert any(check.command == task.pull_checks[0] and check.returncode == 0 for check in result.checks)
    assert any(check.command[0] == "verify-regression" and check.returncode == 0 for check in result.checks)


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
    def trace(*args, **kwargs):
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
        monkeypatch.setattr("github_harness_agent.review_solver.mock_call_diagnostic", lambda *args, **kwargs: "Mock evidence")
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
    "scenario", ("revert", "six_hunks", "mixed", "required_hunk", "low_budget")
)
def test_selective_stock_revert_checks_both_suites(solver_case, monkeypatch, scenario):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    mixed = scenario == "mixed"
    extra_operator_cases = (
        "\ndef test_guard_type():\n    assert isinstance(guard(), int)\n"
        "\ndef test_guard_truthiness():\n    assert not guard()\n"
        if mixed
        else ""
    )
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, guard\n\n"
        "def test_guard():\n    assert guard() == 0\n\n"
        "def test_first():\n    assert first() == -1\n" + extra_operator_cases
    )
    stock_guard = (
        "\ndef test_required_guard():\n    with pytest.raises(ValueError):\n        guard()\n"
        if scenario == "required_hunk"
        else ""
    )
    extra = (30, 40, 50) if scenario in {"six_hunks", "mixed"} else ()
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
            + "\ndef guard() -> int:\n    return 0\n",
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
    guard_edit = (
        '    return "0"\n' if mixed else "    raise ValueError('guard regression')\n"
    )
    proposal = (
        edit(SOURCE, "    return -10\n", left_edit)
        + edit(SOURCE, "    return -20\n", "    return 20\n")
        + edit(SOURCE, "    return 0\n", guard_edit)
        + "".join(
            edit(SOURCE, f"    return -{value}\n", f"    return {value}\n")
            for value in extra
        )
    )
    if mixed:
        mypy = ("mypy", ".")
        task = replace(task, checks=(*task.checks, mypy))
        checked_types = []
        original_command = FixtureWorkspace.command

        def command(workspace, argv, *args, **kwargs):
            if tuple(argv) != mypy:
                return original_command(workspace, argv, *args, **kwargs)
            source = (workspace.root / SOURCE).read_text()
            checked_types.append(source)
            for row, line in enumerate(source.splitlines(), 1):
                if line.strip() == 'return "0"':
                    return SimpleNamespace(
                        returncode=1,
                        stdout=f"./{SOURCE}:{row}: error: Incompatible return value type [return-value]\nFound 1 error in 1 file\n".encode(),
                        stderr=b"",
                    )
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(FixtureWorkspace, "command", command)
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
    rejected = ("+" + guard_edit).encode()
    additions = (*needed, rejected)

    def record(*args, **kwargs):
        failures = evidence(*args, **kwargs)
        patch = args[4]
        if sum(line in patch for line in additions) == len(needed):
            probes.append((patch, failures))
        return failures

    monkeypatch.setattr("github_harness_agent.review_solver.evidence", record)
    if scenario in {"revert", "six_hunks", "mixed"}:
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert all(line in result.patch for line in needed)
        assert rejected not in result.patch
        assert len(gateway.prompts) == 1
        assert result.route_trace[-1].configured_model == "stand-in/free"
        assert probes == [(result.patch, {})]
        if mixed:
            assert any('return "0"' in source for source in checked_types)
            assert checked_types[-2:] == [checked_types[-1]] * 2
            assert 'return "0"' not in checked_types[-1]
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
    "followup",
    (
        "repair",
        "lose_stock",
        "operator_remains",
        "partial_stock",
        "malformed",
        "no_improvement",
        "retry_limit",
        "four_cases",
        "stock_regression_retry",
        "review_regression_retry",
        "static_regression_retry",
    ),
)
def test_private_repair_reconciles_a_stock_gain_without_storing_regressions(
    solver_case, followup, monkeypatch, caplog
):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    caplog.set_level("INFO", logger="github_harness_agent.review_solver")
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
        "def test_second():\n    assert second() == -2\n\n"
        + extra_operator
        + (
            "from fixture_calc import stock_guard\n\n"
            "def test_passing_review_guard():\n    assert stock_guard() == 0\n"
            if followup == "review_regression_retry"
            else ""
        )
    )
    if followup == "static_regression_retry":
        ruff = ("ruff", "check", ".")
        task = replace(task, checks=(*task.checks, ruff))
        original_command = FixtureWorkspace.command

        def command(workspace, argv, *args, **kwargs):
            if tuple(argv) != ruff:
                return original_command(workspace, argv, *args, **kwargs)
            source = (workspace.root / SOURCE).read_text()
            output = b""
            if "unbound_name" in source:
                row = next(
                    row
                    for row, line in enumerate(source.splitlines(), 1)
                    if "unbound_name" in line
                )
                output = f"{SOURCE}:{row}:5: F821 Undefined name `unbound_name`\nFound 1 error.\n".encode()
            return SimpleNamespace(
                returncode=int(bool(output)), stdout=output, stderr=b""
            )

        monkeypatch.setattr(FixtureWorkspace, "command", command)
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: (
                BASE_SOURCE
                + "\ndef stock():\n    return -3\n\ndef stock_guard():\n    return 0\n"
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
        "operator_remains": (
            restore_first,
            edit(SOURCE, "    return 2\n", "    return 4\n"),
        ),
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
        "stock_regression_retry": (
            restore_first + break_stock_guard,
            restore_first,
            restore_second,
        ),
        "review_regression_retry": (
            restore_first + break_stock_guard,
            restore_first,
            restore_second,
        ),
        "static_regression_retry": (
            restore_first
            + edit(SOURCE, "    return -5\n", "    unbound_name\n    return -5\n"),
            restore_first,
            restore_second,
        ),
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

    if followup in {
        "repair",
        "malformed",
        "no_improvement",
        "four_cases",
    } or followup.endswith("_regression_retry"):
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
            assert (
                sum("Private regression repair" in prompt for prompt in gateway.prompts)
                == 3
            )
    else:
        cases = {
            "malformed": ("test_first", "test_first", "test_second"),
            "no_improvement": ("test_first", "test_first", "test_second"),
            "four_cases": ("test_first", "test_second", "test_third", "test_ultimate"),
        }.get(
            followup,
            ("test_first", "test_first", "test_second")
            if followup.endswith("_regression_retry")
            else ("test_first", "test_second"),
        )
        for prompt, case in zip(gateway.prompts[1:], cases, strict=False):
            assert case in prompt.splitlines()[0]
            assert "Private regression repair" in prompt
            assert "test_stock_goal" in prompt
            assert "assert stock() == 3" in prompt
    if followup == "malformed":
        assert after_provider[2] == "provider-2"
        assert "No complete edit blocks" in gateway.prompts[2]
        assert gateway.prompts[2].endswith(
            "Respond with complete named Python definitions only."
        )
    if followup == "no_improvement":
        assert after_provider[2] == "provider-2"
        assert "The selected case still fails" in gateway.prompts[2]
    if followup.endswith("_regression_retry"):
        note = (
            gateway.prompts[2]
            .split("Last proposal feedback for this goal:", 1)[1]
            .split("\nFixture for this goal:", 1)[0]
        )
        assert "New failure from the rejected edit" in note
        if followup == "stock_regression_retry":
            assert "tests/test_existing.py::test_stock_guard" in note
            assert "assert stock_guard() == 0" in note
            assert (
                "(1 stock failures, 1 review failures, 0 static checks)" in caplog.text
            )
        elif followup == "review_regression_retry":
            # The configured operator command executes before the stock command.
            assert "test_passing_review_guard" in note
            assert "assert stock_guard() == 0" in note
            assert "::test_stock_guard" not in note
        else:
            assert "ruff check" in note and "F821" in note and "Undefined name" in note
            assert "> " not in note
            assert (
                "(0 stock failures, 1 review failures, 1 static checks)" in caplog.text
            )
        assert "    unbound_name\n" not in gateway.prompts[2]
    assert len(gateway.prompts) == len(followups) + 1


@pytest.mark.parametrize(
    "scenario",
    (
        "mixed", "static_only", "external", "infra", "unrecognized",
        "output_limit", "preflight_limit",
    ),
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
            if scenario in {"static_only", "output_limit", "preflight_limit"}
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
        if "NOISY_PROPOSAL" in source or scenario == "preflight_limit":
            return original_command(
                workspace,
                (sys.executable, "-c", "print('PRIVATE-OUTPUT' * 10000)"),
                *args,
                **kwargs,
            )
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
    if scenario == "output_limit":
        attempts = (
            edit(SOURCE, '    return "3"\n', '    return 3  # NOISY_PROPOSAL\n'),
            edit(SOURCE, '    return "3"\n', '    return 3\n'),
            edit(
                NEW_TEST,
                "",
                "from fixture_calc import stock\n\ndef test_stock_value():\n    assert stock() == 3\n",
            ),
        )
    exclusions = []

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            exclusions.append(kwargs.get("exclude", ()))
            result = super().complete(**kwargs)
            if scenario == "output_limit":
                result.configured_model = f"model-{len(self.prompts)}"
            return result

    gateway = Gateway(initial, *attempts)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=len(attempts) + 1, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    if scenario == "preflight_limit":
        with pytest.raises(CommandOutputExceeded, match="Command output exceeded"):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert not gateway.prompts and not checkpoints
    elif scenario == "output_limit":
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert b"+    return 3\n" in result.patch and NEW_TEST.encode() in result.patch
        assert b"NOISY_PROPOSAL" not in result.patch
        assert len(gateway.prompts) == 4
        assert exclusions[2] == (("stand-in-free", "model-2"),)
        prompt = gateway.prompts[2]
        assert "Required check 2 (mypy) exceeded the command output limit" in prompt
        assert "unchanged stock-passing draft" in prompt and 'return "3"' in prompt
        assert "NOISY_PROPOSAL" not in prompt and "PRIVATE-OUTPUT" not in prompt
        assert len(checkpoints) == 1 and b"NOISY_PROPOSAL" not in checkpoints[0].patch
    elif good:
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


@pytest.mark.parametrize("scenario", ("count", "later", "loop"))
def test_solver_checkpoints_verified_tracked_assertion_progress(solver_case, scenario):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, second\n\n"
        "def test_first():\n    assert first() == -1\n\n"
        "def test_second():\n    assert second() == -2\n"
    )
    source = (
        BASE_SOURCE
        + "\ndef stock_value():\n    return 5\n\ndef stock_other():\n    return 6\n"
    )
    fixture = "from fixture_calc import stock_value, stock_other\n\ndef test_stock():\n"
    if scenario == "count":
        fixture += "    assert stock_value() == 1\n"
        first_edit = edit(SOURCE, "    return 5\n", "    return 3\n")
        second_edit = edit(SOURCE, "    return 3\n", "    return 1\n")
    else:
        lines = "    assert stock_value() == 1\n    assert stock_other() == 1\n"
        fixture += (
            "    for _ in (None,):\n"
            + "".join("    " + line for line in lines.splitlines(keepends=True))
            if scenario == "loop"
            else lines
        )
        first_edit = edit(SOURCE, "    return 5\n", "    return 1\n")
        second_edit = edit(SOURCE, "    return 6\n", "    return 1\n")
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: source,
            "tests/test_existing.py": fixture,
        }.items():
            body = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(body)
            snapshot.addfile(entry, BytesIO(body))
    gateway = FreeGateway(first_edit, second_edit)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=2, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive.getvalue(), budget, "fixture-version")

    assert all(check.returncode == 0 for check in result.checks)
    assert len(checkpoints) == 1
    assert len(gateway.prompts) == 2
    assert checkpoints[0].progress == (
        "The tracked pytest integer comparison moved closer to its expected value."
        if scenario == "count"
        else "The tracked pytest case reached a later assertion in the same straight-line suite.",
    )


def test_individual_stock_hunks_require_exact_syntax_and_production_scope():
    from github_harness_agent.review_composition import individual_line_edits

    base_text = "VALUE = 1\n# spacer\n# end\n\n"
    base_text += "\n".join(f"BASE_{index} = 0\n" for index in range(4))
    base = {SOURCE: base_text}
    proposal = {
        SOURCE: (
            base_text.replace("VALUE = 1", "VALUE = (")
            .replace("# end", "  1)")
            .replace(" = 0", " = 2")
        )
    }

    variants = list(individual_line_edits(base, proposal, {SOURCE}, limit=3))

    assert variants == [
        {SOURCE: base_text.replace(f"BASE_{index} = 0", f"BASE_{index} = 2")}
        for index in range(3)
    ]
    with_test = {**base, "tests/test_existing.py": ""}
    out_of_scope = {**proposal, "tests/test_existing.py": "raise RuntimeError\n"}
    assert not list(individual_line_edits(with_test, out_of_scope, {SOURCE}, limit=3))


@pytest.mark.parametrize(
    "scenario", ("partial", "focused_pass", "focused_pass_regresses", "static")
)
def test_isolates_stock_hunk_from_regressing_proposal(solver_case, monkeypatch, scenario):
    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    passing_focus = scenario.startswith("focused_pass")
    rejected_focus = scenario == "focused_pass_regresses"
    static_only = scenario == "static"
    mypy = ("mypy", ".")
    if static_only:
        task = replace(task, checks=(*task.checks, mypy))
    operator_assertion = "second() == -2" if static_only else "first() == -1"
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import first, second, stock_value\n\n"
        f"def test_operator():\n    assert {operator_assertion}\n"
        + ("    assert stock_value() == 5\n" if rejected_focus else "")
    )
    stock = "tests/test_existing.py"
    node = f"{stock}::test_stock"
    original_line = "    return 5 if first() == -9 else 5\n"
    if static_only:
        original_line = "    return 5 if not isinstance(first(), int) else 5\n"
    source = BASE_SOURCE.replace("def first():", "def first() -> int:")
    source += "\ndef stock_value():\n" + original_line
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: source,
            stock: (
                "from fixture_calc import stock_value\n\n"
                "def test_stock():\n    assert stock_value() == 1\n\n"
                "def test_other_stock():\n    assert True\n"
            ),
        }.items():
            body = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(body)
            snapshot.addfile(entry, BytesIO(body))

    # The bad hunk masks the result of the second hunk in the full proposal.
    bad_line = '    return "bad"\n' if static_only else "    return -9\n"
    first_hunk = edit(SOURCE, "    return -1\n", bad_line)
    proposed_value = 1 if passing_focus else 3
    isolated_line = original_line.replace("else 5", f"else {proposed_value}")
    proposal = first_hunk + edit(SOURCE, original_line, isolated_line)
    attempts = (
        (proposal,)
        if passing_focus
        else (proposal, edit(SOURCE, isolated_line, "    return 1\n"))
    )

    checks = []
    original_command = FixtureWorkspace.command

    def command(workspace, argv, *args, **kwargs):
        if tuple(argv[1:3]) == ("-m", "pytest"):
            checks.append((tuple(argv), (workspace.root / SOURCE).read_text()))
        if tuple(argv) == mypy:
            checked_source = (workspace.root / SOURCE).read_text()
            checks.append((mypy, checked_source))
            if bad_line in checked_source:
                return SimpleNamespace(
                    returncode=1,
                    stdout=(
                        f"./{SOURCE}:2: error: Incompatible return value type [return-value]\n"
                        "Found 1 error in 1 file\n"
                    ).encode(),
                    stderr=b"",
                )
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return original_command(workspace, argv, *args, **kwargs)

    monkeypatch.setattr(FixtureWorkspace, "command", command)
    gateway = FreeGateway(*attempts)
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=len(attempts), concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    if rejected_focus:
        with pytest.raises(AttemptRejected):
            solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert not checkpoints
    else:
        result = solve(executor, task, archive.getvalue(), budget, "fixture-version")
        assert all(check.returncode == 0 for check in result.checks)
        assert ("+" + bad_line).encode() not in result.patch
        if passing_focus:
            assert not checkpoints
            assert ("+" + isolated_line).encode() in result.patch
        else:
            assert len(checkpoints) == 1
            assert ("+" + isolated_line).encode() in checkpoints[0].patch
            assert ("+" + bad_line).encode() not in checkpoints[0].patch
            assert b"+    return 1\n" in result.patch
    focused = [source for argv, source in checks if argv[-1] == node]
    assert 1 <= len(focused) <= 3
    isolated_source = source.replace(original_line, isolated_line)
    assert isolated_source in focused
    assert all(
        (configured, isolated_source) in checks
        for configured in (*task.pull_checks, *task.checks)
    )


def test_private_stock_repair_keeps_selected_verified_donor_visible(solver_case):
    task, _, runner, _ = solver_case
    task = replace(task, require_new_tests=False)
    Path(task.pull_checks[0][-1]).write_text(
        "from fixture_calc import important as check_value, unrelated\n\n"
        "def test_selected():\n    assert check_value() == 17\n\n"
        "def test_other():\n    assert unrelated() > 0\n"
    )
    prefix = f'PREFIX = "{"a" * 1600}"\n'
    suffix = f'SUFFIX = "{"z" * 1600}"\n'
    source = (
        prefix + BASE_SOURCE + "\ndef stock():\n    return -3\n"
        'def important():\n    return int("17")\n'
        'def unrelated():\n    return int("71")\n' + suffix
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: source,
            "tests/test_existing.py": (
                "from fixture_calc import stock\n\n"
                "def test_stock():\n    assert stock() == 3\n"
            ),
        }.items():
            body = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(body)
            snapshot.addfile(entry, BytesIO(body))
    proposal = (
        edit(SOURCE, prefix, prefix.replace("a" * 1600, "b" * 1600))
        + edit(SOURCE, "    return -3\n", "    return 3\n")
        + edit(SOURCE, '    return int("17")\n', "    return 18\n")
        + edit(SOURCE, '    return int("71")\n', "    return 72\n")
        + edit(SOURCE, suffix, suffix.replace("z" * 1600, "y" * 1600))
    )
    compacted = []

    class Gateway(FreeGateway):
        def complete(self, *, messages, **kwargs):
            if self.prompts:
                compacted.append(compact_focused_review(messages, max_bytes=4096))
            return super().complete(messages=messages, **kwargs)

    gateway = Gateway(
        proposal, edit(SOURCE, "    return 18\n", '    return int("17")\n')
    )
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=2, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive.getvalue(), budget, "fixture-version")

    assert all(check.returncode == 0 for check in result.checks)
    private_prompt = gateway.prompts[1]
    assert "Private regression repair" in private_prompt
    assert 'return int("17")' in private_prompt
    assert 'return int("71")' not in private_prompt
    for prompt in (private_prompt, compacted[0][-1]["content"]):
        assert f"Tentative edit target (exact source): {SOURCE}::important" in prompt
        assert f"Previously verified donor (read-only): {SOURCE}::important" in prompt
        assert "def important():\n    return 18\n" in prompt
        assert 'def important():\n    return int("17")\n' in prompt


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


@pytest.mark.parametrize("late_failure", ("stock", "focused", "setup-only", "stale-module"))
def test_solver_repairs_an_unpublished_regression_test_in_place(
    solver_case, late_failure
):
    task, archive, runner, checkpoints = solver_case
    bad_return = "(0, 0)" if late_failure in {"focused", "stale-module"} else "(1, 2)  # rejected-style"
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
        if late_failure == "stale-module":
            repair = edit(NEW_TEST, bad_test.replace("(0, 0)", "(3, 4)"), TEST)
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


def test_review_fixture_audit_keeps_a_passing_draft_until_missing_coverage_is_added(solver_case):
    task, archive, runner, _ = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    draft = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: BASE_SOURCE, NEW_TEST: ""}, {SOURCE: fixed, NEW_TEST: ""}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    task = replace(
        task,
        checkpoints={draft.key: draft},
        review_recovery=replace(task.review_recovery, feedback=(
            (SOURCE, 2, "Check the first result with a fixture for the first calculation."),
            (SOURCE, 5, "Add a fixture for the second calculation."),
        )),
    )
    incomplete = (
        "from fixture_calc import first, second\n"
        "\ndef test_first_result():\n    assert first() == 1\n"
    )
    repair = (
        incomplete
        + "\ndef test_second_result():\n    assert second() == 2\n"
    )
    first_case = {
        "request": 1,
        "covered": True,
        "case": f"{NEW_TEST}::test_first_result",
        "quote": "assert first() == 1",
    }

    class Gateway(FreeGateway):
        audits = []

        def complete(self, **kwargs):
            if kwargs["role"] != "planning":
                return super().complete(**kwargs)
            request = kwargs["messages"][-1]["content"]
            self.audits.append(request)
            second_case = (
                {
                    "request": 2,
                    "covered": True,
                    "case": f"{NEW_TEST}::test_second_result",
                    "quote": "assert second() == 2",
                }
                if "test_second_result" in request
                else {**first_case, "request": 2, "covered": False}
            )
            return SimpleNamespace(
                content=json.dumps([first_case, second_case]),
                tool_calls=(), role="planning", route="stand-in-free",
                model="fixture-auditor", configured_model="stand-in/auditor", usage={"requests": 1},
            )

    gateway = Gateway(incomplete, "# same case, still missing coverage\n" + incomplete, repair)
    verification = runner.path / "independent"
    verification.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=5, concurrent_probes=1),
        verification_runner=FixtureRunner(verification),
    )
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture-version")

    assert len(gateway.audits) == 2
    assert "with a fixture for the first calculation" in gateway.audits[0]
    assert "fixture for the second calculation" in gateway.audits[0]
    assert "fixture for the second calculation" in gateway.prompts[1]
    assert "Preserve" in gateway.prompts[1]
    assert "fixture for the second calculation" in gateway.prompts[2]
    assert b"def test_first_result" in result.patch
    assert b"def test_second_result" in result.patch
    assert result.usage == {"requests": 2}
    assert result.route_trace[-1].role == "planning"
    assert all(check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize(
    "scenario", ("plain", "fenced", "retry", "named_retry", "accepted_content"),
)
def test_solver_accepts_only_authorized_complete_test_modules(solver_case, scenario):
    task, archive, runner, checkpoints = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    setup = "from fixture_calc import first, second\n" if scenario == "accepted_content" else ""
    draft = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: BASE_SOURCE, NEW_TEST: ""}, {SOURCE: fixed, NEW_TEST: setup}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={draft.key: draft})
    response = f"```python\n{TEST}```" if scenario == "fenced" else TEST
    bad = TEST.replace("(1, 2)", "(0, 0)")
    retry = scenario in {"retry", "named_retry"}
    if scenario == "named_retry":
        bad += "\n\ndef test_passing():\n    assert second() == 2\n"
        _, definition = TEST.split("\n\n", 1)
        response = f"REPLACE {NEW_TEST}::test_results\n{definition}"
    gateway = (
        FreeGateway(edit(NEW_TEST, "", bad), response)
        if retry else FreeGateway(response)
    )
    verification = runner.path / "independent"
    verification.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=2 if retry else 1),
        verification_runner=FixtureRunner(verification),
    )
    budget = _Budget(executor.limits, lambda: False, task)

    if scenario == "accepted_content":
        with pytest.raises(ReviewRoundExhausted):
            solve(executor, task, archive, budget, "fixture-version")
        assert "complete Python module" not in gateway.prompts[0]
        assert not checkpoints
        return
    result = solve(executor, task, archive, budget, "fixture-version")
    assert NEW_TEST.encode() in result.patch and b"(0, 0)" not in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    assert any(NEW_TEST in check.command for check in result.checks)
    assert "complete Python module" in gateway.prompts[0]
    if retry:
        assert bad in gateway.prompts[1] and "complete Python module" in gateway.prompts[1]
    if scenario == "named_retry":
        assert f"REPLACE {NEW_TEST}::" not in gateway.prompts[0]
        assert f"REPLACE {NEW_TEST}::" in gateway.prompts[1]
        assert f"Selected failing case: {NEW_TEST}::test_results" in gateway.prompts[1]
        assert b"def test_passing():" in result.patch
    request = gateway.prompts[-1].replace(
        "\nCurrent checkout:\n", "\nCurrent checkout:\n" + "unused context\n" * 2000
    )
    compact = compact_focused_review(
        [{"role": "system", "content": "Review guidance"}, {"role": "user", "content": request}],
        max_bytes=8192,
    )
    assert compact is not None and "one complete Python module" in compact[-1]["content"]
    assert not checkpoints


@pytest.mark.parametrize("proposal_kind", ("regression", "removed", "collection"))
def test_solver_preserves_verified_passing_cases_in_unpublished_draft(solver_case, proposal_kind):
    task, archive, runner, checkpoints = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    checkpoint = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={checkpoint.key: checkpoint})
    keep = "\n\ndef test_keep():\n    assert second() == 2\n"
    good = TEST + keep
    previous = good.replace("(1, 2)", "(0, 0)")
    if proposal_kind == "regression":
        new_draft = good.replace("second() == 2", "second() == 3")
    elif proposal_kind == "removed":
        new_draft = TEST
    else:
        new_draft = "from fixture_calc import unavailable\n" + previous
    expected_draft = new_draft if proposal_kind == "collection" else previous
    gateway = FreeGateway(
        edit(NEW_TEST, "", previous),
        edit(NEW_TEST, previous, new_draft),
        edit(NEW_TEST, expected_draft, good),
    )
    independent = runner.path / "independent"
    independent.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=3),
        verification_runner=FixtureRunner(independent),
    )

    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture-version")

    assert expected_draft in gateway.prompts[2]
    if proposal_kind == "collection":
        diagnostic = gateway.prompts[2].partition("Latest verification or edit feedback:")[2].partition("Focused source from ")[0]
        assert f"ERROR collecting {NEW_TEST}" in diagnostic
        assert "ImportError" in diagnostic and "unavailable" in diagnostic
    else:
        assert f"Selected failing case: {NEW_TEST}::test_results" in gateway.prompts[2]
        assert f"{NEW_TEST}::test_keep" in gateway.prompts[2]
    assert b"def test_keep():\n+    assert second() == 2" in result.patch
    assert any(NEW_TEST in check.command and check.returncode == 0 for check in result.checks)
    assert not checkpoints


@pytest.mark.parametrize("numbered_reply", ("valid", "stale", "outside", "unchanged"))
def test_solver_retries_failed_pytest_case_with_scoped_numbered_edit(solver_case, numbered_reply):
    task, archive, runner, _ = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    checkpoint = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={checkpoint.key: checkpoint})
    keep = "\n\ndef test_keep():\n    assert second() == 2\n"
    previous = TEST.replace("(1, 2)", "(0, 0)") + keep

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            reply = super().complete(**kwargs)
            if len(self.prompts) != 3:
                return reply
            prompt = self.prompts[-1]
            if match := re.search(r'"snapshot":"([0-9a-f]{64})"', prompt):
                payload = {
                    "snapshot": "0" * 64 if numbered_reply == "stale" else match[1],
                    "edits": [{
                        "start": 3 if numbered_reply == "outside" else 2,
                        "end": 3 if numbered_reply == "outside" else 2,
                        "lines": [
                            "    assert (first(), second()) == "
                            + ("(0, 0)" if numbered_reply == "unchanged" else "(1, 2)")
                        ],
                    }],
                }
                reply.content = "```json\n" + json.dumps(payload) + "\n```"
            return reply

    gateway = Gateway(previous, previous, "No numbered edit supplied.", TEST + keep)
    independent = runner.path / "numbered-verifier"
    independent.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=4),
        verification_runner=FixtureRunner(independent),
    )
    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture")

    prompt = gateway.prompts[2]
    assert prompt.startswith(f"Selected failing case: {NEW_TEST}::test_results\n")
    assert "assert (1, 2) == (0, 0)" in prompt
    assert "2 |     assert (first(), second()) == (0, 0)" in prompt
    assert "def test_keep()" not in prompt
    assert len(gateway.prompts) == (3 if numbered_reply == "valid" else 4)
    assert b"assert (first(), second()) == (1, 2)" in result.patch
    assert b"def test_keep():\n+    assert second() == 2" in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    assert any(NEW_TEST in check.command for check in result.checks)


@pytest.mark.parametrize("inline_setup", (False, True))
def test_repeated_test_format_failures_route_raw_function_to_an_alternate(solver_case, inline_setup):
    task, archive, runner, _ = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    checkpoint = DraftCheckpoint(
        "harness2/base", task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE, last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={checkpoint.key: checkpoint})
    keep = "\n\ndef test_keep():\n    assert second() == 2\n"
    setup = ""
    if inline_setup:
        records = "".join(f"        {step}: {repr('fixture value ' * 8)},\n" for step in range(40))
        setup = "    fake_provider_records = {\n" + records + "    }\n    assert 39 in fake_provider_records\n"
    failed = TEST.replace("    assert", setup + "    assert", 1).replace("(1, 2)", "(0, 0)") + keep
    raw_routes = []

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            if kwargs["tools"]:
                raise ProviderExhausted(None, {"fixture": "no structured route"})
            reply = super().complete(**kwargs)
            if self.prompts[-1].endswith("Return only the complete Python function for this failed test."):
                raw_routes.append(kwargs.get("exclude", ()))
                reply.route, reply.configured_model = f"raw-{len(raw_routes)}", f"raw/{len(raw_routes)}"
            elif len(self.prompts) == 4:
                reply.route, reply.configured_model = "broad-text", "broad/text"
            return reply

    fixed_test = "def test_results():\n" + setup + "    assert (first(), second()) == (1, 2)"
    gateway = Gateway(
        failed, "invalid SEARCH", "invalid numbered edit", "invalid fallback",
        "def test_keep():\n    assert first() == 1", "```python\n" + fixed_test + "\n```",
    )
    independent = runner.path / "raw-test-verifier"
    independent.mkdir()
    executor = PlainExecutor(
        runner, gateway, Limits(model_requests=7, concurrent_probes=1),
        verification_runner=FixtureRunner(independent),
    )
    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture")

    assert len(raw_routes) == 2 and ("raw-1", "raw/1") in raw_routes[1]
    for prompt in gateway.prompts[-2:]:
        assert prompt.startswith(f"Selected failing case: {NEW_TEST}::test_results\n")
        assert "def test_keep():" not in prompt
        assert setup in prompt
    if inline_setup:
        assert len(fixed_test.encode()) > 4096
        assert not any(prompt.endswith(LINE_RANGE_ENDING) for prompt in gateway.prompts)
    assert b"def test_keep():\n+    assert second() == 2" in result.patch
    assert fixed_test.splitlines()[-1].encode() in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    assert any(NEW_TEST in check.command for check in result.checks)


@pytest.mark.parametrize("scenario", ("resume", "different_scope", "projection_fails"))
def test_solver_restores_independently_verified_test_seed(solver_case, scenario):
    task, archive, runner, checkpoints = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    checkpoint = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    saved = []
    task = replace(
        task,
        checkpoints={checkpoint.key: checkpoint},
        save_solver_recovery=lambda state, durable: saved.append((state, durable)),
    )
    if scenario == "resume":
        task = replace(task, checks=((sys.executable, "-m", "pytest", "-q", "tests"),))
    if scenario == "projection_fails":
        require_both = (
            "from pathlib import Path\n"
            f"p = Path({NEW_TEST!r})\n"
            "if p.exists() and 'def test_results(' not in p.read_text():\n"
            "    raise SystemExit('the full test module must contain test_results')\n"
        )
        task = replace(task, checks=(*task.checks, (sys.executable, "-c", require_both)))
    keep = "\n\ndef test_keep():\n    assert second() == 2\n"
    full = TEST + keep
    rejected = full.replace("(1, 2)", "(0, 0)")
    verification = runner.path / "independent"
    verification.mkdir()
    independent = FixtureRunner(verification)

    def run(task, proposals):
        gateway = FreeGateway(*proposals)
        executor = PlainExecutor(
            runner, gateway, Limits(model_requests=len(proposals)),
            verification_runner=independent,
        )
        return solve(
            executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture-version"
        ), gateway

    with pytest.raises(AttemptRejected, match="test_results"):
        run(task, (rejected,))
    if scenario == "projection_fails":
        assert not saved
        return
    [(state, durable)] = saved
    assert durable
    seed = state.test_seed
    assert seed.passing == ("test_keep",)
    assert seed.omitted == ("test_results",)
    assert "def test_keep" in seed.source
    assert "def test_results" not in seed.source
    assert not checkpoints

    restored = SolverRecovery.from_data(state.to_data())
    resumed = replace(task, solver_recovery=restored)
    if scenario == "different_scope":
        result, gateway = run(replace(resumed, checkpoint_scope="another-review"), (full,))
        assert "test_keep" not in gateway.prompts[0]
    else:
        # An edited passing subset still lacks the omitted coverage.
        subset_only = seed.source.replace("second() == 2", "2 == second()")
        result, gateway = run(resumed, (subset_only, full))
        assert len(gateway.prompts) == 2
        assert "def test_keep" in gateway.prompts[0]
        assert "test_results" in gateway.prompts[0]
        assert "Correct the first result." in gateway.prompts[0]
        assert "Correct the second result." in gateway.prompts[0]
    assert b"def test_results()" in result.patch
    assert b"def test_keep()" in result.patch
    assert all(check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize(
    "scenario", (
        "success", "baseline_failure", "no_collection", "ordinary", "protected_anchor",
        "ancestor", "assertion-retention",
    )
)
def test_single_test_only_review_adds_an_independent_module(solver_case, monkeypatch, scenario):
    task, _, runner, checkpoints = solver_case
    task = replace(
        task,
        inline_review_threads=1,
        review_recovery=plan_recovery(
            [{
                "path": "tests/test_existing.py" if scenario == "protected_anchor" else SOURCE,
                "line": 2, "thread": "review", "reply_to": 1,
                "body": (
                    "Please add an offline pytest regression for the reviewed behavior. "
                    "A test-only change in a new test module is sufficient."
                ),
            }],
            0,
            "",
        ),
    )
    if scenario in {"ordinary", "protected_anchor", "ancestor", "assertion-retention"}:
        task = replace(task, pull_checks=())
    archive = BytesIO()
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    existing_test = "def test_prior_coverage():\n    assert True\n"
    prior_setup = (
        "def make_sample():\n    return (4, 2)\n\n"
        "def test_prior_sample():\n    assert make_sample() == (4, 2)\n"
    )
    next_path = "tests/test_review_regressions_2.py"
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        contents = {
            SOURCE: BASE_SOURCE if scenario == "baseline_failure" else fixed,
            "tests/test_existing.py": existing_test,
            NEW_TEST: prior_setup if scenario == "ancestor" else existing_test,
        }
        for path, source in contents.items():
            content = source.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(content)
            snapshot.addfile(entry, BytesIO(content))
    if scenario in {"ancestor", "assertion-retention"}:
        ancestor = BytesIO()
        with tarfile.open(fileobj=ancestor, mode="w") as snapshot:
            data = BASE_SOURCE.encode()
            entry = tarfile.TarInfo(SOURCE)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
        task = replace(task, regression_base="2" * 40, regression_archive=ancestor.getvalue())
        if scenario == "assertion-retention":
            assertion = TEST.replace("(1, 2)", "(0, 0)")
            runtime = assertion.replace("test_results", "test_renamed").replace("first()", "missing()")
            better = assertion.replace("(0, 0)", "(1, 0)")
            proposals = (assertion, runtime, better, edit(next_path, better, TEST))
        else:
            green = "from fixture_calc import first\n\ndef test_results():\n    assert isinstance(first(), int)\n"
            loses_collection = green.replace("import first", "import unavailable")
            proposals = (green, loses_collection, edit(next_path, green, TEST))
    elif scenario == "no_collection":
        proposals = ("__test__ = False\n" + TEST,)
    else:
        proposals = (edit(SOURCE, "return 1\n", "return 3\n"), TEST)
    gateway = FreeGateway(*proposals)
    executor = PlainExecutor(runner, gateway, Limits(model_requests=len(proposals)))
    budget = _Budget(executor.limits, lambda: False, task)
    if scenario == "assertion-retention":
        result = solve(executor, task, archive.getvalue(), budget, "fixture")
        _, _, after_runtime, after_better = gateway.prompts
        assert f"Unpublished test file {next_path} to revise:\n{assertion}" in after_runtime
        assert f"Selected failing case: {next_path}::test_results" in after_runtime
        assert "AssertionError" in after_runtime and "NameError" in after_runtime
        assert runtime not in after_runtime
        assert f"Unpublished test file {next_path} to revise:\n{better}" in after_better
        assert any(check.command[0] == "verify-regression" and not check.returncode for check in result.checks)
        assert not checkpoints
        return
    if scenario == "ancestor":
        result = solve(executor, task, archive.getvalue(), budget, "fixture")
        initial, retry, after_collection = gateway.prompts
        for prompt in (initial, retry, after_collection):
            assert "-    return -1" in prompt and "+    return 1" in prompt
            assert "return -2" not in prompt
            main_prompt = prompt.partition("Optional tracked pytest and production context")[0]
            assert "def make_sample()" in main_prompt
            messages = [{"role": "system", "content": "Use quoted setup."}, {"role": "user", "content": prompt}]
            compact_setup = compact_focused_review(messages, max_bytes=wire_size(messages) - 1)
            assert compact_setup and "def make_sample()" in compact_setup[-1]["content"]
            assert "Optional tracked pytest and production context" not in compact_setup[-1]["content"]
            optional = prompt.partition("Optional tracked pytest and production context")[2]
            assert "def first():\n    return 1" in optional
        assert "together in one test execution" in initial
        for prompt in (retry, after_collection):
            lead = prompt.partition("Edit only")[0]
            assert "The new tests also pass on the ancestor." in lead
            compact = compact_focused_review(
                [{"role": "system", "content": "Use quoted source data."},
                 {"role": "user", "content": prompt}], max_bytes=1600,
            )
            assert compact and "-    return -1" in compact[-1]["content"]
            assert "The new tests also pass on the ancestor." in compact[-1]["content"]
        assert "ImportError" in after_collection
        assert f"Unpublished test file {next_path} to revise:\n{green}" in after_collection
        assert f"+++ b/{next_path}".encode() in result.patch
        assert any(check.command[0] == "verify-regression" and not check.returncode for check in result.checks)
        return
    if scenario == "protected_anchor":
        rollouts = []

        def rollout(task, archive, budget, prompt, role, *, checkpoint_key, **kwargs):
            rollouts.append(checkpoint_key)
            before = {SOURCE: fixed, next_path: ""}
            production = fixed + "\nunused_constant = 3\n" if len(rollouts) == 1 else fixed
            after = {SOURCE: production, next_path: TEST}
            assert "Add only new test files." in prompt
            return patch_for(before, after), "", None

        monkeypatch.setattr(executor, "_rollout", rollout)
        result = verify_review_repair(executor, task, archive.getvalue(), budget)
        assert rollouts == ["harness2/base", "harness2/repair"]
        assert SOURCE.encode() not in result.patch
        assert f"+++ b/{next_path}".encode() in result.patch
        assert any(next_path in check.command for check in result.checks)
        return
    if scenario == "baseline_failure":
        with pytest.raises(ExecutionError, match="unchanged baseline"):
            verify_review_repair(executor, task, archive.getvalue(), budget)
        assert not gateway.prompts
    elif scenario == "no_collection":
        with pytest.raises(AttemptRejected):
            verify_review_repair(executor, task, archive.getvalue(), budget)
        assert len(gateway.prompts) == 1
    else:
        result = verify_review_repair(executor, task, archive.getvalue(), budget)
        assert len(gateway.prompts) == 2
        assert next_path in gateway.prompts[0]
        assert [line for line in result.patch.splitlines() if line.startswith((b"--- ", b"+++ "))] == [
            b"--- /dev/null", f"+++ b/{next_path}".encode(),
        ]
        assert all(check.returncode == 0 for check in result.checks)
        assert any(next_path in check.command for check in result.checks)
        assert all(any(command == check.command for check in result.checks) for command in (
            *task.checks, *task.pull_checks
        ))
    assert not checkpoints


@pytest.mark.parametrize("alternatives_exhausted", (False, True))
def test_test_authoring_rotates_after_repeated_checked_failure_and_reopens_routes(
    solver_case, alternatives_exhausted,
):
    task, archive, runner, _ = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    draft = DraftCheckpoint(
        "harness2/base", task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE, last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={draft.key: draft})
    cheap = ("stand-in-free", "stand-in/free")
    exclusions = []
    passing = "\n\ndef test_keep_existing_success():\n    assert second() == 2\n"
    partial = TEST.replace("(1, 2)", "(0, 0)") + passing
    final = TEST + passing

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            excluded = kwargs.get("exclude", ())
            exclusions.append(excluded)
            if len(exclusions) == 3:
                assert excluded == (cheap,)
                assert kwargs["after_provider"] == cheap[0]
                if alternatives_exhausted:
                    raise ProviderExhausted(None, {"fixture": "all eligible models excluded"})
                reply = super().complete(**kwargs)
                reply.route, reply.configured_model = "alternate-free", "alternate/free"
                return reply
            return super().complete(**kwargs)

    bad = (
        "from fixture_calc import missing_a\n\n"
        "def test_results():\n    assert missing_a() == 1\n"
    )
    proposals = (bad, bad.replace("missing_a", "missing_b"))
    if not alternatives_exhausted:
        proposals += (partial,)
    gateway = Gateway(*proposals, final)
    executor = PlainExecutor(runner, gateway, Limits(model_requests=4, concurrent_probes=1))
    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture")

    assert exclusions == [(), (), (cheap,), ()]
    if not alternatives_exhausted:
        assert "def test_keep_existing_success():" in gateway.prompts[-1]
    assert b"def test_keep_existing_success():" in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    assert any(NEW_TEST in check.command for check in result.checks)


@pytest.mark.parametrize(
    "tool_reply", ("valid", "extra", "duplicate", "unavailable", "empty", "failing_test"),
)
def test_solver_recovers_malformed_new_test_replies_with_one_scoped_tool(solver_case, tool_reply):
    task, archive, runner, checkpoints = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    draft = DraftCheckpoint(
        "harness2/base", task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE, last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={draft.key: draft})
    calls = []

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            tools = kwargs["tools"]
            calls.append(bool(tools))
            if len(calls) > 2:
                assert kwargs["exclude"] == (("stand-in-free", "stand-in/free"),)
            if not tools:
                reply = super().complete(**kwargs)
                if len(calls) > 2:
                    reply.route, reply.configured_model = "text-free", "fixture-text"
                return reply
            schema, = tools
            name = schema["function"]["name"]
            assert set(schema["function"]["parameters"]["properties"]) == {"code"}
            assert NEW_TEST in kwargs["messages"][-1]["content"]
            if tool_reply == "unavailable":
                raise ProviderExhausted(None, {"test": "no tool-capable routes"})
            bad = TEST.replace("(1, 2)", "(0, 0)")
            arguments = {"code": bad if tool_reply == "failing_test" else TEST}
            if tool_reply == "extra":
                arguments["path"] = SOURCE
            encoded = json.dumps(arguments)
            if tool_reply == "duplicate":
                encoded = encoded[:-1] + ', "code": ' + json.dumps(TEST) + "}"
            native_call = {"type": "function", "function": {"name": name, "arguments": encoded}}
            return SimpleNamespace(
                content="", tool_calls=() if tool_reply == "empty" else (native_call,),
                role="coding", route="tool-free", model="fixture-tool",
                configured_model="fixture-tool", usage={},
            )

    gateway = Gateway("This is malformed.", "Still no edit.", TEST)
    executor = PlainExecutor(runner, gateway, Limits(model_requests=4, concurrent_probes=1))
    result = solve(executor, task, archive, _Budget(executor.limits, lambda: False, task), "fixture")
    assert calls == ([False, False, True] if tool_reply == "valid" else [False, False, True, False])
    assert all(NEW_TEST.encode() not in checkpoint.patch for checkpoint in checkpoints)
    assert NEW_TEST.encode() in result.patch and b"(0, 0)" not in result.patch
    assert all(check.returncode == 0 for check in result.checks)
    assert any(NEW_TEST in check.command for check in result.checks)
    if tool_reply == "failing_test":
        assert "test_results" in gateway.prompts[-1] and "(0, 0)" in gateway.prompts[-1]


def test_solver_reserves_one_native_test_reply_per_unpublished_revision(solver_case):
    task, archive, runner, checkpoints = solver_case
    fixed = BASE_SOURCE.replace("return -1", "return 1").replace("return -2", "return 2")
    checkpoint = DraftCheckpoint(
        "harness2/base",
        task.base_sha,
        patch_for({SOURCE: BASE_SOURCE}, {SOURCE: fixed}),
        guidance=_GUIDANCE,
        last_editor=("fixture", "fixture/coder"),
    )
    task = replace(task, checkpoints={checkpoint.key: checkpoint})
    first = TEST.replace("(1, 2)", "(0, 0)")
    second = TEST.replace("(1, 2)", "(1, 0)")
    calls = []
    recovery = iter(((first, second), (second, first)))

    class Gateway(FreeGateway):
        def complete(self, **kwargs):
            tools = kwargs["tools"]
            calls.append(bool(tools))
            if not tools:
                return super().complete(**kwargs)
            prompt = kwargs["messages"][-1]["content"]
            expected, replacement = next(recovery)
            assert expected in prompt
            tool, = tools
            return SimpleNamespace(
                content="",
                tool_calls=({
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "arguments": json.dumps({"code": replacement}),
                    },
                },),
                role="coding", route="tool-free", model="fixture-tool",
                configured_model="fixture-tool", usage={},
            )

    gateway = Gateway(first, *(["Malformed edit."] * 6), TEST)
    executor = PlainExecutor(runner, gateway, Limits(model_requests=12, concurrent_probes=1))
    budget = _Budget(executor.limits, lambda: False, task)

    result = solve(executor, task, archive, budget, "fixture")

    assert calls == [False, False, False, True, False, False, True, False, False, False]
    assert first in gateway.prompts[-1]
    assert b"assert (first(), second()) == (1, 2)" in result.patch
    assert any(NEW_TEST in check.command and check.returncode == 0 for check in result.checks)
    assert not checkpoints


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


@pytest.mark.parametrize(
    ("failure", "fresh_sparse"),
    [
        ("malformed", False),
        ("nonimproving", False),
        ("success", True),
        ("sparse_stale", True),
        ("sparse_nonimproving", True),
        ("sparse_outage", True),
    ],
)
def test_localized_stock_repairs_choose_numbered_and_named_formats(
    solver_case, monkeypatch, failure, fresh_sparse
):
    from github_harness_agent import review_solver
    from github_harness_agent.providers import ProviderExhausted

    task, _, runner, _ = solver_case
    task = replace(task, require_new_tests=False, excluded_models=("blocked/model",))
    archive = BytesIO()
    stock_method = (
        "def stock_result():\n"
        + (
            "    # preserve the existing method body and comments\n" * 125
            if fresh_sparse
            else ""
        )
        + "    return -3\n"
    )
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
                if failure == "sparse_outage":
                    raise ProviderExhausted(None, {"local": "upstream timeout"})
                row = int(re.search(r"(?m)^([0-9]+) \|     return -3$", prompt)[1])
                snapshot = re.search(r"\"snapshot\":\"([a-f0-9]{64})\"", prompt)[1]
                if failure == "sparse_stale":
                    snapshot = "0" * 64
                replacement = (
                    "    return -8"
                    if failure == "sparse_nonimproving"
                    else "    return 3"
                )
                content = json.dumps(
                    {
                        "snapshot": snapshot,
                        "edits": [{"start": row, "end": row, "lines": [replacement]}],
                    }
                )
            elif prompt.endswith(
                "Respond with complete named Python definitions only."
            ):
                content = (
                    "bad named response"
                    if failure == "malformed"
                    else f"REPLACE {SOURCE}::stock_result\n"
                    + stock_method.replace(
                        "return -3", "return 3" if fresh_sparse else "return -8"
                    )
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

    if fresh_sparse:
        expected = ["coding"] * 3 + ["planning", "coding"]
        if failure != "success":
            expected.append("coding")
            assert calls[-1][2].endswith(
                "Respond with complete named Python definitions only."
            )
        assert [item[0] for item in calls] == expected
        _, system, prompt, request, max_tokens = calls[4]
    else:
        assert [item[0] for item in calls] == ["coding"] * 3 + ["planning"] + [
            "coding"
        ] * 3
        assert calls[4][2].endswith("Respond with exact SEARCH/REPLACE blocks only.")
        assert calls[5][2].endswith(
            "Respond with complete named Python definitions only."
        )
        _, system, prompt, request, max_tokens = calls[6]
    assert "single line-range JSON object" in prompt
    assert "1 | def stock_result():" in prompt
    assert f"{len(stock_method.splitlines())} |     return -3" in prompt
    assert prompt.count(stock_method) == 1
    assert "Complete pinned baseline definition" in prompt
    assert "causal evidence" in prompt and "Trace the public inputs." in prompt
    assert "assert stock_result() == 3" in prompt
    assert "Review notes" in prompt
    assert "SEARCH" not in system and "REPLACE" not in system
    assert "bad named response" not in prompt and "return -8" not in prompt
    assert request.get("after_provider") is None
    assert request.get("exclude") == ("blocked/model",)
    assert max_tokens == (3072 if fresh_sparse else 6144)
    assert system.endswith(review_solver.CAUSAL_EDIT_GUIDANCE)
    assert all(check.returncode == 0 for check in result.checks)
    assert (
        b"+    return 3\n" in result.patch
        and b"tests/test_existing.py" not in result.patch
    )

@pytest.mark.parametrize(
    ("interactive_tests", "scenario"),
    [
        (False, "localized"),
        (True, "localized"),
        (False, "outage"),
        (False, "turn_cap"),
        (True, "turn_cap"),
        (False, "outage_regression"),
        (True, "outage_partial_retry"),
        (False, "outage_missing_tests_retry"),
        (False, "outage_unrequested_tests"),
        (False, "unlocalized"),
        (False, "long_evidence"),
        (False, "one_attempt"),
        (False, "wrong_snapshot"),
        (False, "reserved"),
    ],
)
def test_saved_stock_plateau_tries_scoped_tools_then_verifies_regression(
    solver_case, monkeypatch, caplog, interactive_tests, scenario
):
    from github_harness_agent import review_solver
    from github_harness_agent.providers import ProviderExhausted

    caplog.set_level("INFO", logger=review_solver.__name__)
    task, _, runner, checkpoints = solver_case
    recovery = []
    long_evidence = scenario == "long_evidence"
    outage = scenario.startswith("outage")
    turn_cap = scenario == "turn_cap"
    partial_retry = scenario == "outage_partial_retry"
    retry_after_outage = partial_retry or scenario == "outage_missing_tests_retry"
    unrequested_tests = scenario == "outage_unrequested_tests"
    if unrequested_tests:
        task = replace(task, require_new_tests=False)
    localized = scenario == "localized" or long_evidence or outage or turn_cap
    acceptance = Path(task.pull_checks[0][-1])
    with acceptance.open("a") as file:
        file.write("\nfrom fixture_calc import stock_result\n"
                   "\ndef test_stock_baseline():\n    assert stock_result(baseline=True) == 7\n")
    task = replace(task, instructions=task.instructions + "\n" + "Original review background. " * 600)
    task = replace(task, save_solver_recovery=lambda state, durable: recovery.append((state, durable)))
    if interactive_tests:
        task = replace(task, checks=((sys.executable, "-m", "pytest", "-q", "tests"),))
    initial_source = BASE_SOURCE + (
        "\ndef stock_result(*, baseline=False):\n"
        "    if baseline:\n        return 7\n"
        "    return -3\n"
    )
    test_body = "from fixture_calc import stock_result\n\ndef test_stock_result(tracked_stock_seed):\n"
    fixture_body = "import pytest\n\n@pytest.fixture\ndef tracked_stock_seed():\n"
    if long_evidence:
        initial_source += "".join(f"    unused_value_{row} = {row}\n" for row in range(150))
        test_body += "".join(f"    checked_input_{row} = {row}\n" for row in range(160))
        fixture_body += "".join(f"    fixture_input_{row} = {row}\n" for row in range(55))
    assertion_line = len(test_body.splitlines()) + 1
    expectation = "3" if partial_retry else "tracked_stock_seed"
    test_body += f"    assert stock_result() == {expectation}\n"
    fixture_body += "    return int('03')\n"
    if scenario == "wrong_snapshot":
        target = (1, "tests/test_existing.py::test_stock_result")
        saved_source = initial_source.replace("return -1", "return 1").replace("return -2", "return 2")
        checkpoint = DraftCheckpoint(
            review_solver._CHECKPOINT, task.base_sha,
            patch_for({SOURCE: initial_source}, {SOURCE: saved_source}), guidance=_GUIDANCE,
        )
        task = replace(
            task, checkpoints={checkpoint.key: checkpoint},
            solver_recovery=SolverRecovery("0" * 64, ((review_solver._goal_digest(target), 2, 2),)),
        )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, content in {
            SOURCE: initial_source,
            "tests/conftest.py": fixture_body,
            "tests/test_existing.py": test_body,
        }.items():
            data = content.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            snapshot.addfile(entry, BytesIO(data))
    traces = ""
    if long_evidence:
        traces = (
            "Auxiliary mapping counter changes (quoted advisory data). "
            "The pinned case passed and the draft failed at its numeric assertion. "
            "Locations show where the probe observed each change, with reviewed callers.\n"
            f"Pinned: 0 -> 7 after {SOURCE}:8 stock_result; " + "more pinned context " * 15 + "\n"
            f"Draft: 0 -> -3 after {SOURCE}:10 stock_result; " + "more draft context " * 15 + "\n"
            "Auxiliary mock calls (quoted advisory data)\n"
            "The pinned base passed and the current draft failed. "
            "Each call is [method, arguments, keyword arguments].\n"
            "lookup on base [1]\nlookup on current [3]"
        )
    monkeypatch.setattr(review_solver, "mock_call_diagnostic", lambda *args, **kwargs: traces)
    monkeypatch.setattr(
        review_solver, "stock_guard_diagnostic",
        lambda *args: "Auxiliary counterfactual (quoted data, partial diagnostic only). fixture_calc.py::stock_result, draft line 8: causal evidence.",
    )
    early_success = scenario in {"one_attempt", "wrong_snapshot"}
    fix = edit(SOURCE, "    return -3\n", "    return 3\n")
    text_edits = iter((
        *((FIRST + SECOND,) if scenario != "wrong_snapshot" else ()),
        edit(SOURCE, "    return -3\n", "    return -4\n"),
        fix if early_success else edit(SOURCE, "    return -3\n", "    return -5\n"),
    ))
    phases = []
    tool_turns = []
    regression = "from fixture_calc import stock_result\n\ndef test_repair():\n    assert stock_result() == 3\n"
    rejected_checks = []
    if scenario == "outage_regression":
        def record(*args, **kwargs):
            result = evidence(*args, **kwargs)
            if b"+        return 8\n" in args[4]:
                rejected_checks.append(result)
            return result

        monkeypatch.setattr(review_solver, "evidence", record)

    focused_command = None

    def call(name, arguments):
        return {"id": name + arguments.get("path", ""), "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    async def mock_wire(request):
        nonlocal focused_command
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=local_inventory())
        assert scenario in {"localized", "unlocalized", "long_evidence"} or outage or turn_cap
        payload = json.loads(request.content)
        calls = ()
        tools, messages = payload["tools"], payload["messages"]
        tool_turns.append(payload["max_tokens"])
        phases.append("interactive")
        outputs = [message["content"] for message in messages if message["role"] == "tool"]
        task_prompt = next(m["content"] for m in messages if m["role"] == "user")
        if long_evidence:
            assert wire_size(messages) <= 8192
            assert f"tests/test_existing.py:{assertion_line}" in task_prompt
            assert f"(diagnostic at line {assertion_line})" in task_prompt
            assert "assert stock_result() == tracked_stock_seed" in task_prompt
            assert "assert -3 == 3" in task_prompt
            assert "Auxiliary counterfactual" in task_prompt
            assert f"Pinned: 0 -> 7 after {SOURCE}:8" in task_prompt
            assert f"Draft: 0 -> -3 after {SOURCE}:10" in task_prompt
            assert "lookup on current [3]" in task_prompt
        if not outputs:
            if not long_evidence:
                assert f"def test_stock_result(tracked_stock_seed):\n    assert stock_result() == {expectation}" in task_prompt
                assert "def tracked_stock_seed():\n    return int('03')" in task_prompt
            assert "def test_stock_baseline():\n    assert stock_result(baseline=True) == 7" in task_prompt
            command_text = re.search(r"(?m)^Focused pytest argv: (.+)$", task_prompt)
            assert command_text
            focused_command = json.loads(command_text[1])
            assert focused_command[-1] == "tests/test_existing.py::test_stock_result"
            assert "Localized method in the current draft" in task_prompt
            if not long_evidence:
                assert "def stock_result(*, baseline=False):\n    if baseline:\n        return 7\n    return -3" in task_prompt
            assert {tool["function"]["name"] for tool in tools} >= {"run", "edit_file"}, phases
            calls = (
                call("edit_file", {"path": "outside.py", "new_text": "bad = True\n"}),
                call("run", {"argv": ["sed", "-n", "1,10p", SOURCE]}),
            )
        elif len(tool_turns) == 2:
            assert "outside the scoped review" in outputs[0]
            assert "return 1" in outputs[1] and "return 2" in outputs[1]
            assert "return -3" in outputs[1]
            if scenario == "outage_regression":
                old, new = "        return 7\n    return -3\n", "        return 8\n    return 3\n"
            elif turn_cap:
                old, new = "    return -3\n", "    return 0\n"
            else:
                old, new = "    return -3\n", "    return -1\n" if partial_retry else "    return 3\n"
            calls = (call("edit_file", {"path": SOURCE, "old_text": old, "new_text": new}),)
            if interactive_tests or unrequested_tests:
                calls += (call("edit_file", {"path": NEW_TEST, "new_text": regression}),)
        elif len(tool_turns) == 3:
            if unrequested_tests:
                assert "outside the scoped review" in outputs[-1]
            calls = (call("run", {"argv": focused_command}),)
        elif turn_cap:
            calls = (
                (call("edit_file", {"path": SOURCE, "old_text": "    return 0\n", "new_text": "    return 3\n"}),)
                if len(tool_turns) == 12
                else (call("run", {"argv": ["cat", SOURCE]}),)
            )
        else:
            assert "1 passed" in outputs[-1]
        message = {"tool_calls": calls} if calls else {"content": "done"}
        return httpx.Response(200, json={"model": LOCAL_MODEL, "choices": [{"message": message}]})

    mock_async_local(monkeypatch, mock_wire)
    wire = routed([pinned_local(local_input_bytes=8192)], lambda _: pytest.fail('non-local wire'))

    class Gateway:
        def complete(self, *, role, messages, tools, timeout, **kwargs):
            prompt = messages[-1]["content"]
            calls = ()
            if retry_after_outage and "provider outage" in phases:
                raise ProviderExhausted(None, {"fixture": "waiting for provider retry"})
            if tools:
                if scenario == "outage_regression" and {tool["function"]["name"] for tool in tools} == {"replace_python_definition"}:
                    raise ProviderExhausted(None, {"fixture": "native tool unavailable"})
                if outage and len(tool_turns) == 3:
                    phases.append("provider outage")
                    raise ProviderExhausted(None, {"fixture": "temporary outage"})
                return wire.complete(role=role, messages=messages, tools=tools, timeout=timeout, **kwargs)
            if role == "planning":
                phases.append("planning")
                content = json.dumps({"strategy": "Trace the public inputs.", "symbols": [f"{SOURCE}::stock_result" if localized else "unknown.py::not_reviewed"]})
            elif prompt.startswith("Selected failure: new regression tests"):
                phases.append("author")
                if scenario in {"localized", "unlocalized", "long_evidence", "outage"}:
                    assert len(checkpoints) == 2
                    assert b"+    return 3\n" not in checkpoints[0].patch
                    assert b"+    return 3\n" in checkpoints[-1].patch
                    assert NEW_TEST.encode() not in checkpoints[-1].patch
                content = edit(NEW_TEST, "", regression)
            elif "single line-range JSON object" in prompt:
                phases.append("numbered")
                content = "invalid line range"
            elif scenario == "reserved" and phases[-1:] == ["guided search"]:
                phases.append("text")
                content = fix
            elif scenario == "outage_regression" and "provider outage" in phases:
                phases.append("text")
                assert "return -3" in prompt and "return 8" not in prompt
                content = "An incomplete edit"
            elif prompt.endswith("Respond with complete named Python definitions only."):
                phases.append("named")
                content = "invalid named edit"
            elif "Trace the public inputs." in prompt or (not localized and phases[-1:] == ["planning"]):
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
    request_limit = 7 if scenario == "reserved" else 24 if turn_cap else 14
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=request_limit, concurrent_probes=1),
        verification_runner=FixtureRunner(verifier_root),
    )
    budget = _Budget(executor.limits, lambda: False, task)
    if retry_after_outage:
        with pytest.raises((ProviderExhausted, ReviewRoundExhausted)):
            solve(executor, task, archive.getvalue(), budget, "fixture")
        initial, saved = checkpoints
        assert initial.patch != saved.patch and NEW_TEST.encode() not in saved.patch
        assert "provider outage" in phases and "author" not in phases
        if partial_retry:
            assert b"+    return -1\n" in saved.patch
            assert saved.progress == (
                "The tracked pytest integer comparison moved closer to its expected value.",
            )
        else:
            assert b"+    return 3\n" in saved.patch
        resumed = replace(task, checkpoints={saved.key: saved})
        edits = (
            [edit(SOURCE, "    return -1\n", "    return 3\n")] if partial_retry else []
        ) + [edit(NEW_TEST, "", regression)]
        resumed_gateway = FreeGateway(*edits)
        resumed_executor = PlainExecutor(
            runner,
            resumed_gateway,
            Limits(model_requests=len(edits), concurrent_probes=1),
            verification_runner=executor.verification_runner,
        )
        resumed_budget = _Budget(resumed_executor.limits, lambda: False, resumed)
        result = solve(
            resumed_executor, resumed, archive.getvalue(), resumed_budget, "fixture"
        )
        first_goal = "test_stock_result" if partial_retry else "new regression tests"
        assert first_goal in resumed_gateway.prompts[0].splitlines()[0]
        assert all(check.returncode == 0 for check in result.checks)
        assert b"+    return 3\n" in result.patch and NEW_TEST.encode() in result.patch
        return
    if scenario == "outage_regression":
        with pytest.raises(ReviewRoundExhausted):
            solve(executor, task, archive.getvalue(), budget, "fixture")
        assert len(rejected_checks) == 1
        assert {name.rsplit("::", 1)[-1] for _, name in rejected_checks[0]} == {"test_stock_baseline"}
        assert all(b"+        return 8\n" not in checkpoint.patch for checkpoint in checkpoints)
        assert "provider outage" in phases and "author" not in phases
        return
    result = solve(executor, task, archive.getvalue(), budget, "fixture")

    expected = ["text"] * (2 if scenario == "wrong_snapshot" else 3)
    if not early_success:
        expected += ["planning", "guided search"]
        if localized:
            expected += ["named", "numbered"]
        if outage:
            expected += ["interactive"] * 3 + ["provider outage"]
        elif turn_cap:
            expected += ["interactive"] * 12
        else:
            expected += ["text"] if scenario == "reserved" else ["interactive"] * 4
        assert any(durable for _, durable in recovery)
    else:
        assert not any(durable for _, durable in recovery)
    if not interactive_tests and not unrequested_tests:
        expected.append("author")
    assert phases == expected
    if scenario == "localized" and not interactive_tests:
        assert "(exact edits succeeded=1 rejected=1, shell actions=2)" in caplog.text
    if turn_cap:
        assert (
            f"(exact edits succeeded={2 + int(interactive_tests)} rejected=1, shell actions=10)"
            in caplog.text
        )
    if scenario == "reserved":
        assert budget.model_requests_remaining() == 0
    if scenario != "wrong_snapshot":
        assert b"+    return 3\n" not in checkpoints[0].patch
    assert all(0 < tokens <= 3072 for tokens in tool_turns)
    assert b"+    return 3\n" in result.patch
    assert (NEW_TEST.encode() in result.patch) is not unrequested_tests
    assert all(NEW_TEST.encode() not in checkpoint.patch for checkpoint in checkpoints)
    assert b"outside.py" not in result.patch
    assert all(check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize("outage", (False, True))
def test_scoped_rollout_rejects_unreviewed_files_written_through_run(solver_case, outage):
    from github_harness_agent.providers import ProviderExhausted

    task, archive, runner, _ = solver_case
    task = replace(task, save_checkpoint=None, review_recovery=None, require_new_tests=False)

    class Gateway:
        def complete(self, *, role, messages, **request):
            finished = any(message["role"] == "tool" for message in messages)
            if finished and outage:
                raise ProviderExhausted(None)
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
            if outage:
                commands = ({"id": "edit", "type": "function", "function": {
                    "name": "edit_file", "arguments": json.dumps({
                        "path": SOURCE, "old_text": "    return -1\n", "new_text": "    return 1\n",
                    }),
                }}, *commands)
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


@pytest.mark.parametrize(
    "scenario", ("attributed", "no_edit", "cancelled", "expired", "unattributed")
)
@pytest.mark.parametrize("stop", ("outage", "request_cap", "turn_cap"))
def test_scoped_outage_requires_live_budget_and_attributable_source(
    solver_case, scenario, stop
):
    from github_harness_agent.executor import AttemptBudgetRejected
    from github_harness_agent.providers import ProviderExhausted
    from github_harness_agent.review_solver import _InteractiveBudget
    from github_harness_agent.worker import Cancelled, LimitExceeded

    task, archive, runner, _ = solver_case
    task = replace(
        task, save_checkpoint=None, review_recovery=None, require_new_tests=False
    )
    cancellation = Event()
    turns = []

    class Gateway:
        def complete(self, *, role, **request):
            turns.append(role)
            if len(turns) == (3 if stop == "outage" else 2):
                if scenario == "cancelled":
                    cancellation.set()
                if scenario == "expired":
                    parent.deadline = 0
                if stop == "outage":
                    raise ProviderExhausted(None)
            if len(turns) == 1 and scenario != "no_edit":
                name, params = (
                    "edit_file",
                    {
                        "path": SOURCE,
                        "old_text": "    return -1\n",
                        "new_text": "    return 1\n",
                    },
                )
                model = "editor"
            else:
                argv = ["cat", SOURCE]
                if scenario == "unattributed":
                    argv = [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; p = Path({SOURCE!r}); p.write_text(p.read_text() + '# shell-only edit\\n')",
                    ]
                name, params = "run", {"argv": argv}
                model = "observer"
            call = {
                "id": str(len(turns)),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(params)},
            }
            return SimpleNamespace(
                role=role,
                route=model,
                model=model,
                configured_model=model,
                content=None,
                tool_calls=(call,),
                usage={},
            )

    class RacingBudget(_Budget):
        def complete(self, *args, **kwargs):
            if stop == "request_cap" and len(turns) == 2:
                # A concurrent request acquires the final slot after the rollout's
                # remaining-request check and before the delegated request claims it.
                self.claim("model request", self.limits.model_requests)
            return super().complete(*args, **kwargs)

    executor = PlainExecutor(runner, Gateway(), Limits(model_requests=3))
    parent = RacingBudget(executor.limits, cancellation.is_set, task)
    budget = parent if stop == "outage" else _InteractiveBudget(
        parent, 2 if stop == "turn_cap" else 3, 8
    )
    scope = {SOURCE: BASE_SOURCE}
    if scenario == "attributed":
        patch, _, actor = executor._rollout(
            task,
            archive,
            budget,
            "Scoped tool calls.",
            "coding",
            clarification=False,
            scoped_sources=scope,
        )
        assert b"+    return 1\n" in patch
        assert "    return 1\n" in scope[SOURCE]
        assert actor.configured_model == "editor"
    else:
        expected = {"cancelled": Cancelled, "expired": LimitExceeded}.get(
            scenario, ProviderExhausted if stop == "outage" else AttemptBudgetRejected
        )
        with pytest.raises(expected) as caught:
            executor._rollout(
                task,
                archive,
                budget,
                "Scoped tool calls.",
                "coding",
                clarification=False,
                scoped_sources=scope,
            )
        assert type(caught.value) is expected
        assert scope == {SOURCE: BASE_SOURCE}
    if stop != "outage":
        assert len(turns) == 2


@pytest.mark.parametrize(
    "scenario", ("outside", "protected", "shell_test", "tests_only", "tool_limit", "unmatched", "mode")
)
def test_scoped_turn_cap_refuses_side_effects_and_tests_without_production(
    solver_case, scenario
):
    from github_harness_agent.review_solver import _InteractiveBudget

    task, archive, runner, _ = solver_case
    task = replace(task, save_checkpoint=None, review_recovery=None)
    path = {
        "outside": "outside.py",
        "protected": "tests/test_existing.py",
        "shell_test": NEW_TEST,
        "tests_only": NEW_TEST,
    }.get(scenario, SOURCE)
    production = (
        "edit_file",
        {
            "path": SOURCE,
            "old_text": "    return -9\n" if scenario == "unmatched" else "    return -1\n",
            "new_text": "    return 1\n",
        },
    )
    if scenario == "tests_only":
        actions = [("edit_file", {"path": path, "new_text": "def test_result():\n    assert True\n"})]
    elif scenario in {"tool_limit", "unmatched"}:
        actions = [production, ("run", {"argv": ["cat", SOURCE]})]
    elif scenario == "mode":
        actions = [production, ("run", {"argv": ["chmod", "+x", SOURCE]})]
    else:
        script = f"from pathlib import Path; Path({path!r}).write_text('pass\\n')"
        actions = [production, ("run", {"argv": [sys.executable, "-c", script]})]

    class Gateway:
        def complete(self, *, role, **request):
            name, params = actions.pop(0)
            call = {
                "id": name,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(params)},
            }
            return SimpleNamespace(
                role=role, route="editor", model="editor", configured_model="editor",
                content=None, tool_calls=(call,), usage={},
            )

    executor = PlainExecutor(runner, Gateway(), Limits(model_requests=3))
    parent = _Budget(executor.limits, lambda: False, task)
    probe = _InteractiveBudget(parent, len(actions), 1 if scenario == "tool_limit" else 8)
    scope = {SOURCE: BASE_SOURCE, NEW_TEST: ""}
    with pytest.raises(AttemptRejected):
        executor._rollout(
            task, archive, probe, "Scoped tool calls.", "coding",
            clarification=False, scoped_sources=scope,
        )
    assert scope == {SOURCE: BASE_SOURCE, NEW_TEST: ""}


@pytest.mark.parametrize(
    "scenario", ("direct", "stale", "stale_twice", "route_outage", "unscoped")
)
def test_scoped_rollout_paces_exact_edit_then_restores_verification(
    solver_case, scenario
):
    from collections import Counter

    from github_harness_agent.providers import ProviderExhausted

    task, archive, runner, _ = solver_case
    task = replace(
        task, save_checkpoint=None, review_recovery=None, require_new_tests=False
    )
    ordinary = {"run", "edit_file"}
    inspect_source = ("run", {"argv": ["cat", SOURCE]})
    exact = (
        "edit_file",
        {"path": SOURCE, "old_text": "    return -1\n", "new_text": "    return 1\n"},
    )
    verification = ("run", {"argv": list(task.checks[0])})
    attempts = [(ordinary, inspect_source), (ordinary, inspect_source)]
    if scenario in {"stale", "stale_twice"}:
        stale = ("edit_file", {**exact[1], "old_text": "    return stale\n"})
        attempts += [({"edit_file"}, stale), ({"run"}, inspect_source)]
        if scenario == "stale_twice":
            attempts += [({"edit_file"}, stale), (ordinary, exact)]
        else:
            attempts.append(({"edit_file"}, exact))
    elif scenario == "route_outage":
        attempts += [({"edit_file"}, None), (ordinary, exact)]
    else:
        attempts.append((ordinary if scenario == "unscoped" else {"edit_file"}, exact))
    attempts += [(ordinary, verification), (ordinary, ("finish", {}))]
    used = []

    class Gateway:
        def complete(self, *, role, messages, tools, **request):
            names, action = attempts[len(used)]
            assert {tool["function"]["name"] for tool in tools} == names
            used.append(names)
            if action is None:
                raise ProviderExhausted(None)
            name, arguments = action
            if names == {"edit_file"}:
                assert "exact" in messages[0]["content"].lower()
                assert SOURCE in messages[0]["content"]
                assert any(
                    "return -1" in str(item.get("content")) for item in messages[1:]
                )
            if names == {"run"}:
                assert "old_text must occur exactly once" in messages[-1]["content"]
            calls = (
                ()
                if name == "finish"
                else (
                    {
                        "id": str(len(used)),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    },
                )
            )
            return SimpleNamespace(
                role=role,
                route="fixture",
                model="fixture",
                configured_model="fixture",
                content="done" if not calls else None,
                tool_calls=calls,
                usage={},
            )

    executor = PlainExecutor(runner, Gateway(), Limits(model_requests=len(attempts)))
    budget = _Budget(executor.limits, lambda: False, task)
    scope = None if scenario == "unscoped" else {SOURCE: BASE_SOURCE}
    counts = Counter()
    patch, _, actor = executor._rollout(
        task,
        archive,
        budget,
        "Test scoped pacing.",
        "coding",
        clarification=False,
        scoped_sources=scope,
        scoped_action_counts=counts,
    )

    assert b"+    return 1\n" in patch
    assert len(used) == len(attempts)
    assert actor.configured_model == "fixture"
    if scope is not None:
        assert "    return 1\n" in scope[SOURCE]
        assert counts["successful_edits"] == 1
        assert counts["rejected_edits"] == {"stale": 1, "stale_twice": 2}.get(
            scenario, 0
        )
        assert counts["shell_actions"] == (4 if scenario.startswith("stale") else 3)
    else:
        assert not counts


@pytest.mark.parametrize("scenario", ("same", "changed_selection", "resume", "rotate"))
def test_pinned_stock_counterfactual_only_feeds_verified_private_repair(
    solver_case, monkeypatch, scenario
):
    from github_harness_agent import review_solver
    from github_harness_agent.worker import Cancelled

    task, _, runner, checkpoints = solver_case
    different_selection = scenario == "changed_selection"
    resume = scenario == "resume"
    rotate = scenario == "rotate"
    dependency_seed = resume or rotate
    states = []
    lease_ended = Event()

    def save_recovery(state, durable):
        states.append((state, durable))
        if resume and durable and state and state.alternate:
            lease_ended.set()

    task = replace(
        task,
        checks=((sys.executable, "-m", "pytest", "-q", "tests"),),
        require_new_tests=False,
        save_solver_recovery=save_recovery,
    )
    acceptance = Path(task.pull_checks[0][-1])
    with acceptance.open("a") as stream:
        stream.write(
            "\nfrom fixture_calc import stock_value\n"
            "\ndef test_review_value():\n    assert stock_value(review=True) == 8\n"
        )
        if dependency_seed:
            stream.write(
                "\ndef test_review_z_extra():\n    assert stock_value(review='extra') == 80\n"
            )
    stock_return = "    return floor(3.5)\n" if dependency_seed else "    return 3\n"
    method = (
        "\ndef stock_value(*, review=False):\n"
        + ("    if review == 'extra':\n        return 70\n" if dependency_seed else "")
        + "    if review:\n"
        "        return 7\n"
        + "    # preserve the existing definition and its lengthy comments\n" * 110
        + stock_return
    )
    original = (
        ("from math import floor\n\n" if dependency_seed else "") + BASE_SOURCE + method
    )
    test = (
        "from fixture_calc import stock_value\n"
        "import pytest\n\n"
        "def test_stock():\n    assert stock_value() == 3\n"
    )
    if different_selection:
        test += (
            '\n@pytest.mark.parametrize("value", [None], ids=[str(stock_value())])\n'
            "def test_additional(value):\n    assert value is None\n"
        )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, text in {SOURCE: original, "tests/test_stock.py": test}.items():
            data = text.encode()
            member = tarfile.TarInfo(path)
            member.size = len(data)
            snapshot.addfile(member, BytesIO(data))

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
            f"{SOURCE}::stock_value, draft line {line}: causal evidence."
        )

    monkeypatch.setattr(review_solver, "stock_guard_diagnostic", localize)
    monkeypatch.setattr(
        review_solver, "mock_call_diagnostic", lambda *args, **kwargs: ""
    )
    probes = []
    actual_collect = review_solver.stock_collection

    def collect(*args, **kwargs):
        fingerprint = actual_collect(*args, **kwargs)
        probes.append(fingerprint)
        return fingerprint

    monkeypatch.setattr(review_solver, "stock_collection", collect)
    prompts = []
    extra = (
        edit(SOURCE, "        return 70\n", "        return 80\n")
        if dependency_seed
        else ""
    )
    remove_import = (
        edit(SOURCE, "from math import floor\n\n", "") if dependency_seed else ""
    )
    fix_first = edit(SOURCE, "        return 7\n", "        return 8\n")
    private_edits = iter(
        (
            fix_first + edit(SOURCE, stock_return, "    return 0\n"),
            edit(SOURCE, "        return 7\n", "        return 9\n"),
            extra,
            fix_first,
        )
    )
    early_edits = iter(
        (
            FIRST
            + SECOND
            + edit(SOURCE, "        return 7\n", "        return 8\n")
            + extra
            + remove_import
            + edit(SOURCE, stock_return, "    return -3\n"),
            edit(
                SOURCE,
                "def stock_value(*, review=False):\n",
                "# first stall\ndef stock_value(*, review=False):\n",
            ),
            edit(
                SOURCE,
                "def stock_value(*, review=False):\n",
                "# second stall\ndef stock_value(*, review=False):\n",
            ),
        )
    )

    class Gateway:
        def complete(self, *, role, messages, **kwargs):
            prompt = messages[-1]["content"]
            prompts.append(prompt)
            if role == "planning":
                content = json.dumps(
                    {
                        "strategy": "Trace the public modes.",
                        "symbols": [f"{SOURCE}::stock_value"],
                    }
                )
            elif "Private regression repair." in prompt:
                assert checkpoints and b"+    return -3\n" in checkpoints[-1].patch
                assert stock_return.strip() in prompt
                content = next(private_edits) if rotate else fix_first
            elif "Advisory strategy" in prompt:
                content = "unusable format"
            elif len(prompts) <= 3:
                content = next(early_edits)
            elif "Respond with complete named Python definitions only." in prompt:
                content = f"REPLACE {SOURCE}::stock_value\n" + method.replace(
                    "return 7", "return 8"
                )
            else:
                content = edit(SOURCE, "    return -3\n", "    return 3\n")
            return SimpleNamespace(
                role=role,
                route="fixture-free",
                model="fixture-coder",
                configured_model="fixture/free",
                content=content,
                tool_calls=(),
                usage={},
            )

    executor = PlainExecutor(
        runner,
        Gateway(),
        Limits(model_requests=13 if rotate else 10, concurrent_probes=1),
    )
    budget = _Budget(executor.limits, lease_ended.is_set, task)
    if resume:
        with pytest.raises(Cancelled):
            solve(executor, task, archive.getvalue(), budget, "fixture")
        saved = [
            state for state, durable in states if durable and state and state.alternate
        ]
        assert len(saved) == 1
        assert len(checkpoints) == 1
        assert b"+    return 3\n" not in checkpoints[-1].patch
        assert len(saved[0].alternate.cases) == 1
        assert len(saved[0].alternate.preceding_cases) == 2
        assert saved[0].alternate.dependency_seed
        assert b"+        return 8\n" in saved[0].alternate.patch
        resumed = replace(
            task,
            checkpoints={checkpoints[-1].key: checkpoints[-1]},
            solver_recovery=SolverRecovery.from_data(saved[0].to_data()),
        )
        gateway = FreeGateway(extra)
        executor = PlainExecutor(
            runner, gateway, Limits(model_requests=1, concurrent_probes=1)
        )
        fresh = _Budget(executor.limits, lambda: False, resumed)
        result = solve(executor, resumed, archive.getvalue(), fresh, "fixture")
        assert "Private regression repair." in gateway.prompts[0]
        assert "test_review_z_extra" in gateway.prompts[0].splitlines()[0]
        assert all(check.returncode == 0 for check in result.checks)
        assert b"+        return 80\n" in result.patch
        assert len(probes) == 6 and all(probe == probes[0] for probe in probes)
        assert states[-1][0] is None or states[-1][0].alternate is None

        alternate = saved[0].alternate
        wrong_case = next(iter(set(alternate.preceding_cases) - set(alternate.cases)))
        for changed in (
            replace(alternate, identity="f" * 64),
            replace(alternate, cases=(wrong_case,)),
            replace(alternate, collection="f" * 64),
        ):
            stale = replace(
                resumed, solver_recovery=replace(saved[0], alternate=changed)
            )
            gateway = FreeGateway("incomplete edit")
            executor = PlainExecutor(
                runner, gateway, Limits(model_requests=1, concurrent_probes=1)
            )
            with pytest.raises(AttemptRejected):
                solve(
                    executor,
                    stale,
                    archive.getvalue(),
                    _Budget(executor.limits, lambda: False, stale),
                    "fixture",
                )
            assert "Private regression repair." not in gateway.prompts[0]
            assert not any(state and state.alternate for state, _ in states[-1:])
        return
    result = solve(executor, task, archive.getvalue(), budget, "fixture")
    assert all(check.returncode == 0 for check in result.checks)
    private = [prompt for prompt in prompts if "Private regression repair." in prompt]
    if rotate:
        assert [prompt.splitlines()[0].rsplit("::", 1)[-1] for prompt in private] == [
            "test_review_value",
            "test_review_value",
            "test_review_z_extra",
            "test_review_value",
        ]
        assert "New failure from the rejected edit" in private[1]
        assert "tests/test_stock.py::test_stock" in private[1]
        assert "Last proposal feedback for this goal" not in private[2]
        assert "assert stock_value(review='extra') == 80" in private[2]
        assert b"+        return 80\n" in result.patch
        assert b"+    return 0\n" not in result.patch
        return
    assert len(private) == (0 if different_selection else 1)
    assert len(probes) == (2 if different_selection else 3)
    if different_selection:
        assert all(probes) and probes[0] != probes[1]
    else:
        assert all(probe == probes[0] for probe in probes) and probes[0]
    assert b"tests/test_stock.py" not in result.patch
    assert b"+        return 8\n" in result.patch


@pytest.mark.parametrize(
    "scenario", ["one_request", "planner_outage", "untyped", "rejected_seed"]
)
def test_resumed_stock_plateau_can_try_pinned_restore_before_planning(
    solver_case, monkeypatch, scenario
):
    from hashlib import sha256

    from github_harness_agent import review_solver
    from github_harness_agent.providers import ProviderExhausted
    from github_harness_agent.review_mock_replay import (
        CounterCallerLocation,
        CounterCallerSite,
        _ReplayDiagnostic,
    )

    task, _, runner, checkpoints = solver_case
    task = replace(task, require_new_tests=False)
    acceptance = Path(task.pull_checks[0][-1])
    with acceptance.open("a") as stream:
        stream.write(
            "\nfrom fixture_calc import stock_value\n"
            "\ndef test_review_value():\n    assert stock_value(review=True) == 8\n"
        )
    original = BASE_SOURCE + (
        "\ndef stock_value(*, review=False):\n"
        "    if review:\n"
        "        return 7\n"
        + "    # preserve ordinary reviewed method behavior\n" * 140
        + "    return 3\n"
    )
    current = (
        original.replace("return -1", "return 1")
        .replace("return -2", "return 2")
        .replace("return 7", "return 8")
        .replace("return 3", "return -3")
    )
    patch = patch_for({SOURCE: original}, {SOURCE: current})
    draft = DraftCheckpoint(
        review_solver._CHECKPOINT,
        task.base_sha,
        patch,
        guidance=_GUIDANCE,
        last_editor=("fixture-free", "fixture/free"),
    )
    target = (1, "tests/test_existing.py::test_stock")
    task = replace(
        task,
        checkpoints={draft.key: draft},
        solver_recovery=SolverRecovery(
            review_solver._recovery_digest(
                task, {acceptance.name: acceptance.read_text()}, patch
            ),
            ((review_solver._goal_digest(target), 0, 2),),
        ),
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, text in {
            SOURCE: original,
            "tests/test_existing.py": (
                "from fixture_calc import stock_value\n"
                "\ndef test_stock():\n    assert stock_value() == 3\n"
            ),
        }.items():
            data = text.encode()
            member = tarfile.TarInfo(path)
            member.size = len(data)
            snapshot.addfile(member, BytesIO(data))

    calls = {"replay": 0, "guard": 0, "seed": 0}
    real_seed = review_solver.pinned_stock_restore_seed

    def replay(*args, **kwargs):
        calls["replay"] += 1
        row = current.splitlines().index("    return -3") + 1
        site = CounterCallerSite(
            (
                CounterCallerLocation(
                    SOURCE, "stock_value", row, sha256(current.encode()).hexdigest()
                ),
            ),
            "Pinned and draft caller evidence.",
        )
        text = "Auxiliary mapping counter changes (quoted advisory data)."
        return text if scenario == "untyped" else _ReplayDiagnostic(text, site)

    def guard(*args, **kwargs):
        calls["guard"] += 1
        return ""

    def seed(*args, **kwargs):
        calls["seed"] += 1
        prepared = real_seed(*args, **kwargs)
        if scenario == "rejected_seed":
            return replace(
                prepared,
                advisory="Auxiliary pinned dependencies (quoted advisory data). Synthetic binding.",
            )
        return prepared

    monkeypatch.setattr(review_solver, "mock_call_diagnostic", replay)
    monkeypatch.setattr(review_solver, "stock_guard_diagnostic", guard)
    monkeypatch.setattr(review_solver, "pinned_stock_restore_seed", seed)
    if scenario == "rejected_seed":
        monkeypatch.setattr(
            review_solver, "stock_collection", lambda *args, **kwargs: None
        )
    phases = []
    prompts = []

    class Gateway:
        def complete(self, *, role, messages, **kwargs):
            phases.append(role)
            prompt = messages[-1]["content"]
            prompts.append(prompt)
            if role == "planning":
                raise ProviderExhausted(None, {"planner": "upstream timeout"})
            content = (
                edit(SOURCE, "        return 7\n", "        return 8\n")
                if "Private regression repair." in prompt
                else edit(SOURCE, "    return -3\n", "    return 3\n")
            )
            return SimpleNamespace(
                role=role,
                route="fixture-free",
                model="fixture-coder",
                configured_model="fixture/free",
                content=content,
                usage={},
                tool_calls=(),
            )

    request_cap = 1 if scenario == "one_request" else 3
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=request_cap, concurrent_probes=1)
    )
    budget = _Budget(executor.limits, lambda: False, task)
    result = solve(executor, task, archive.getvalue(), budget, "fixture")
    assert all(check.returncode == 0 for check in result.checks)
    assert not checkpoints
    assert calls["replay"] == 1
    if scenario in {"one_request", "planner_outage"}:
        assert phases == ["coding"]
        assert calls["seed"] == 1 and calls["guard"] == 0
        assert "Private regression repair." in prompts[0]
        assert "test_review_value" in prompts[0]
    else:
        assert phases == ["planning", "coding"]
        assert calls["guard"] == 1
        assert calls["seed"] == int(scenario == "rejected_seed")
        assert all("Private regression repair." not in prompt for prompt in prompts)
        assert prompts[0].count("Synthetic binding.") == int(
            scenario == "rejected_seed"
        )


@pytest.mark.parametrize("scenario", ["verified", "spoof", "unrelated", "regression", "escape", "after_seed"])
def test_resumed_repeated_stock_uses_only_typed_scoped_coding(solver_case, monkeypatch, scenario):
    from github_harness_agent import review_solver
    from github_harness_agent.providers import ProviderExhausted
    from github_harness_agent.review_mock_replay import (
        CounterCallerSite,
        _ReplayDiagnostic,
        counter_caller_site,
        counter_loop_reuse,
    )

    task, _, runner, checkpoints = solver_case
    acceptance = Path(task.pull_checks[0][-1])
    with acceptance.open("a") as stream:
        stream.write(
            "\nfrom fixture_calc import account_for\n"
            "\ndef test_distinct_records():\n"
            '    assert account_for([{"reference": "a"}, {"reference": "b"}])["encountered"] == 2\n'
        )
    mutation = '        result["encountered"] += 1'
    before = BASE_SOURCE + (
        "\ndef account_for(records):\n"
        "    seen = set()\n"
        '    result = {"encountered": 0}\n'
        + "".join(
            "    # Unrelated middle body to keep out of the localized prompt.\n"
            if index == 16 else "    # Unchanged processing notes.\n"
            for index in range(185 if scenario == "after_seed" else 35)
        )
        + "    for item in records[:1]:\n"
        '        reference = item["reference"]\n'
        + mutation + "\n"
        "    return result\n"
    )
    current = (
        before.replace("return -1", "return 1").replace("return -2", "return 2")
        .replace("records[:1]", "records")
    )
    tracked = (
        "from fixture_calc import account_for\n\n"
        "def test_stock():\n"
        '    item = {"reference": "a"}\n'
        '    assert account_for([item, item])["encountered"] == 1\n'
    )
    patch = patch_for({SOURCE: before}, {SOURCE: current})
    target = (1, "tests/test_existing.py::test_stock")
    draft = DraftCheckpoint(
        review_solver._CHECKPOINT, task.base_sha, patch, guidance=_GUIDANCE,
        last_editor=("checkpoint-free", "fixture/checkpoint"),
    )
    task = replace(
        task, checkpoints={draft.key: draft},
        solver_recovery=SolverRecovery(
            review_solver._recovery_digest(task, {acceptance.name: acceptance.read_text()}, patch),
            ((review_solver._goal_digest(target), 0, 2),),
        ),
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, source in {SOURCE: before, "tests/test_existing.py": tracked}.items():
            data = source.encode()
            member = tarfile.TarInfo(path)
            member.size = len(data)
            snapshot.addfile(member, BytesIO(data))

    actual_replay = review_solver.mock_call_diagnostic
    traces = []

    def replay(*args, **kwargs):
        report = actual_replay(*args, **kwargs)
        signal = counter_loop_reuse(report)
        assert signal and signal.entries == 2 and signal.pinned_mutations == 1
        traces.append(signal)
        if scenario == "spoof":
            return str(report)
        if scenario == "after_seed":
            return _ReplayDiagnostic(
                report, CounterCallerSite((signal.owner,), "Paired caller evidence."), signal
            )
        if scenario == "unrelated":
            return _ReplayDiagnostic(
                report, counter_caller_site(report),
                replace(signal, node="tests/test_another.py::test_other"),
            )
        return report

    monkeypatch.setattr(review_solver, "mock_call_diagnostic", replay)
    monkeypatch.setattr(review_solver, "stock_guard_diagnostic", lambda *args: "")
    phases = []
    replacement = [
        "        if reference not in seen:",
        "            seen.add(reference)",
        '            result["encountered"] += 1',
    ]

    class Gateway:
        def complete(self, *, role, messages, timeout, **options):
            prompt = messages[-1]["content"]
            if role == "planning":
                phases.append("planning")
                raise ProviderExhausted(None, {"planner": "upstream timeout"})
            if "Repair repeated logical processing with a small scoped edit" in prompt:
                phases.append("precision")
                assert 0 < timeout <= 35
                assert "after_provider" in options and options["after_provider"] == "checkpoint-free"
                assert 'assert account_for([item, item])["encountered"] == 1' in prompt
                assert "across 2 loop entries" in prompt
                assert "Unrelated middle body" not in prompt
                assert "for item in records:" in prompt and "seen = set()" in prompt
                row = int(re.search(r"(?m)^(\d+) \| " + re.escape(mutation) + "$", prompt)[1])
                digest = re.search(r'"snapshot":"([a-f0-9]{64})"', prompt)[1]
                change = replacement
                if scenario == "regression":
                    change = ["        if not seen:", *replacement[1:]]
                elif scenario == "escape":
                    row -= 18  # Hidden middle body, outside all editable numbered windows.
                content = json.dumps({"snapshot": digest, "edits": [{"start": row, "end": row, "lines": change}]})
                route = "precision-free"
            elif "Private regression repair." in prompt:
                phases.append("private")
                content = "malformed"
                route = "private-free"
            elif prompt.startswith("Selected failure: new regression tests"):
                phases.append("author")
                content = edit(NEW_TEST, "", tracked.replace("test_stock", "test_repeated"))
                route = "regression-author"
            else:
                phases.append("ordinary")
                content = edit(SOURCE, mutation + "\n", "\n".join(replacement) + "\n")
                route = "ordinary-free"
            return SimpleNamespace(
                role=role, route=route, model="fixture-coder", configured_model=route,
                content=content, usage={}, tool_calls=(),
            )

    verification = runner.path / "verify"
    verification.mkdir()
    executor = PlainExecutor(
        runner, Gateway(), Limits(model_requests=8, concurrent_probes=1),
        verification_runner=FixtureRunner(verification),
    )
    result = solve(executor, task, archive.getvalue(), _Budget(executor.limits, lambda: False, task), "fixture")
    expected = ["precision"] if scenario not in {"spoof", "unrelated"} else []
    if scenario == "after_seed":
        expected = ["private"] * 3 + expected
    elif scenario != "verified":
        expected += ["planning", "ordinary"]
    assert phases == [*expected, "author"]
    assert len(traces) == 1
    assert checkpoints and checkpoints[-1].last_editor[0] == (
        "precision-free" if scenario in {"verified", "after_seed"} else "ordinary-free"
    )
    assert all(b"if not seen:" not in checkpoint.patch for checkpoint in checkpoints)
    assert b"if reference not in seen:" in result.patch
    assert b"tests/test_existing.py" not in result.patch
    assert NEW_TEST.encode() in result.patch
    assert all(check.returncode == 0 for check in result.checks)
