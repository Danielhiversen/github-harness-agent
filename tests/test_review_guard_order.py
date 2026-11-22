import hashlib
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from test_executor import FixtureRunner
from test_incremental_solver import BASE_SOURCE, NEW_TEST, SOURCE, edit
from test_incremental_solver import solver_case as build_solver_case

from github_harness_agent import review_solver
from github_harness_agent.checkpoint import DraftCheckpoint, SolverRecovery
from github_harness_agent.executor import (
    AttemptRejected,
    Limits,
    PlainExecutor,
    _Budget,
)
from github_harness_agent.providers import ProviderExhausted
from github_harness_agent.review_guard_order import swap_traced_filters
from github_harness_agent.review_mock_replay import (
    CounterCallerLocation,
    CounterLoopReuse,
    counter_loop_reuse,
)

solver_case = build_solver_case

FIRST = """            if self.primary and not self.primary.allows(a, b):
                self.record("primary")
                continue
"""
SECOND = """            if not gate.allows(a, b):
                self.record("secondary")
                continue
"""
SOURCE_TEXT = (
    BASE_SOURCE
    + """
class Policy:
    def allows(self, a, b):
        return a == b


class Processor:
    def __init__(self):
        self.primary = Policy()
        self.reasons = {"primary": 0, "secondary": 0}

    def record(self, reason):
        self.reasons[reason] += 1

    def run(self, rows, gate):
        a, b = 1, 2
        for entry in rows:
"""
    + FIRST
    + SECOND
    + """        return self.reasons


def process(rows):
    return Processor().run(rows, Policy())
"""
)
TARGET = (1, "tests/test_existing.py::test_stock")


def observation(source):
    lines = source.splitlines()
    digest = hashlib.sha256(source.encode()).hexdigest()
    location = CounterCallerLocation(
        SOURCE,
        "Processor.run",
        lines.index('                self.record("primary")') + 1,
        digest,
    )
    mutation = CounterCallerLocation(
        SOURCE,
        "Processor.record",
        lines.index("        self.reasons[reason] += 1") + 1,
        digest,
    )
    return CounterLoopReuse(
        location,
        mutation,
        lines.index("        for entry in rows:") + 2,
        "entry",
        2,
        2,
        0,
        TARGET[1],
        5,
    )


def test_traced_filter_swap_moves_only_the_two_intact_guards():
    scope = {SOURCE: SOURCE_TEXT, "unchanged.py": "sentinel = True\n"}
    expected = SOURCE_TEXT.replace(FIRST + SECOND, SECOND + FIRST)
    assert swap_traced_filters(scope, {SOURCE}, observation(SOURCE_TEXT)) == {
        **scope,
        SOURCE: expected,
    }
    assert scope[SOURCE] == SOURCE_TEXT


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "            if not gate",
            "            marker = True\n            if not gate",
        ),
        (
            "            if not gate",
            "            # Second policy\n            if not gate",
        ),
        ("gate.allows(a, b)", "gate.allows(a, c)"),
        ("gate.allows(a, b)", "gate.permits(a, b)"),
        ("gate.allows(a, b)", "gate.allows(a, call())"),
        (
            '                self.record("secondary")',
            '                entry.append("secondary")',
        ),
        (
            '                self.record("secondary")',
            '                gate.record("secondary")',
        ),
        (
            '                self.record("secondary")',
            "                value = compute()",
        ),
        (SECOND, SECOND.replace("continue", "break")),
        (SECOND, SECOND + "            else:\n                marker = True\n"),
    ),
)
def test_traced_filter_swap_rejects_unsafe_or_nonadjacent_guards(old, new):
    source = SOURCE_TEXT.replace(old, new)
    assert swap_traced_filters({SOURCE: source}, {SOURCE}, observation(source)) is None


def test_traced_filter_swap_requires_current_scoped_repeat_evidence():
    sources = {SOURCE: SOURCE_TEXT}
    trace = observation(SOURCE_TEXT)
    assert swap_traced_filters(sources, set(), trace) is None
    assert swap_traced_filters(sources, {SOURCE}, None) is None
    assert swap_traced_filters(sources, {SOURCE}, replace(trace, entries=1)) is None
    assert (
        swap_traced_filters(sources, {SOURCE}, replace(trace, variable="other")) is None
    )
    assert (
        swap_traced_filters(
            sources,
            {SOURCE},
            replace(trace, owner=replace(trace.owner, line=trace.owner.line - 1)),
        )
        is None
    )
    assert (
        swap_traced_filters(
            sources,
            {SOURCE},
            replace(trace, mutation=replace(trace.mutation, name="Processor.run")),
        )
        is None
    )
    assert swap_traced_filters({SOURCE: SOURCE_TEXT + "\n"}, {SOURCE}, trace) is None


def test_traced_filter_swap_rejects_a_sink_on_the_shared_iterable():
    source = SOURCE_TEXT.replace('self.record("', 'rows.append("')
    trace = observation(SOURCE_TEXT)
    owner = replace(
        trace.owner, source_digest=hashlib.sha256(source.encode()).hexdigest()
    )
    trace = replace(trace, owner=owner, mutation=owner)
    assert swap_traced_filters({SOURCE: source}, {SOURCE}, trace) is None


@pytest.mark.parametrize("scenario", ("pass", "review_regression", "no_new_tests"))
def test_resumed_guard_swap_requires_fresh_review_and_model_tests(
    solver_case, monkeypatch, scenario
):
    task, _, runner, checkpoints = solver_case
    acceptance = Path(task.pull_checks[0][-1])
    if scenario == "review_regression":
        with acceptance.open("a") as stream:
            stream.write(
                '\nfrom fixture_calc import process\n\ndef test_distinct_rows():\n    assert process([{}, {}])["primary"] == 2\n'
            )
    base = SOURCE_TEXT.replace(FIRST + SECOND, SECOND + FIRST)
    current = SOURCE_TEXT.replace("return -1", "return 1").replace(
        "return -2", "return 2"
    )
    fixed = current.replace(FIRST + SECOND, SECOND + FIRST)
    tracked = (
        "from fixture_calc import process\n\n"
        "def test_stock():\n    item = {}\n"
        '    assert process([item, item])["primary"] == 0\n'
    )
    patch = review_solver.patch_for({SOURCE: base}, {SOURCE: current})
    fixed_patch = review_solver.patch_for({SOURCE: base}, {SOURCE: fixed})
    draft = DraftCheckpoint(
        review_solver._CHECKPOINT,
        task.base_sha,
        patch,
        guidance=review_solver._GUIDANCE,
        last_editor=("checkpoint-free", "fixture/checkpoint"),
    )
    task = replace(
        task,
        checkpoints={draft.key: draft},
        solver_recovery=SolverRecovery(
            review_solver._recovery_digest(
                task, {acceptance.name: acceptance.read_text()}, patch
            ),
            ((review_solver._goal_digest(TARGET), 0, 2),),
        ),
    )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, source in {SOURCE: base, "tests/test_existing.py": tracked}.items():
            data = source.encode()
            member = tarfile.TarInfo(path)
            member.size = len(data)
            snapshot.addfile(member, BytesIO(data))

    actual_replay = review_solver.mock_call_diagnostic
    actual_evidence = review_solver.evidence
    traces = []
    checks = []

    def replay(*args, **kwargs):
        report = actual_replay(*args, **kwargs)
        if signal := counter_loop_reuse(report):
            traces.append(signal)
        return report

    def evidence(*args, **kwargs):
        checked = actual_evidence(*args, **kwargs)
        if args[4] == fixed_patch:
            checks.append((args[5], checked))
        return checked

    monkeypatch.setattr(review_solver, "mock_call_diagnostic", replay)
    monkeypatch.setattr(review_solver, "evidence", evidence)
    tasks = []

    class Gateway:
        def complete(self, *, role, messages, **options):
            selection = messages[-1]["content"].splitlines()[0]
            tasks.append(selection)
            if selection != "Selected failure: new regression tests":
                raise ProviderExhausted(None, {"fixture": "not available"})
            test = (
                "from fixture_calc import first, process\n\n"
                "def test_reordered_processing_preserves_review():\n"
                "    assert first() == 1\n"
                '    assert process([{}, {}])["secondary"] == 2\n'
            )
            return SimpleNamespace(
                role=role,
                route="model-test-author",
                model="fixture",
                configured_model="fixture",
                content=edit(NEW_TEST, "", test)
                if scenario == "pass"
                else "incomplete test",
                tool_calls=(),
                usage={},
            )

    verifier = runner.path / "verification"
    verifier.mkdir()
    executor = PlainExecutor(
        runner,
        Gateway(),
        Limits(model_requests=3, concurrent_probes=1),
        verification_runner=FixtureRunner(verifier),
    )
    budget = _Budget(executor.limits, lambda: False, task)
    if scenario == "pass":
        result = review_solver.solve(
            executor, task, archive.getvalue(), budget, "fixture"
        )
        assert all(check.returncode == 0 for check in result.checks)
        assert NEW_TEST.encode() in result.patch
    else:
        with pytest.raises((AttemptRejected, ProviderExhausted)):
            review_solver.solve(executor, task, archive.getvalue(), budget, "fixture")
    assert traces and traces[0].entries == 2
    assert traces[0].owner.name == "Processor.run"
    assert traces[0].mutation.name == "Processor.record"
    assert checks[0][0][0][-1] == TARGET[1] and not checks[0][1]
    assert checks[1][0] == (*task.pull_checks, *task.checks)
    if scenario == "review_regression":
        assert any("test_distinct_rows" in case for _, case in checks[1][1])
        assert "Selected failure: new regression tests" not in tasks
    else:
        assert not checks[1][1]
        assert tasks and set(tasks) == {"Selected failure: new regression tests"}
    assert not checkpoints
