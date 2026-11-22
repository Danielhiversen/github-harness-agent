import tarfile
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from test_executor import FixtureRunner
from test_incremental_solver import NEW_TEST, SOURCE, FreeGateway, edit
from test_incremental_solver import solver_case as build_solver_case
from test_review_guard_order import FIRST, SECOND, SOURCE_TEXT

from github_harness_agent import review_solver
from github_harness_agent.checkpoint import DraftCheckpoint
from github_harness_agent.executor import (
    AttemptRejected,
    Limits,
    PlainExecutor,
    _Budget,
)
from github_harness_agent.providers import ProviderExhausted

solver_case = build_solver_case


@pytest.mark.parametrize(
    "scenario", ("accept", "trade_cases", "renamed_stock", "static_regression")
)
def test_guard_order_with_existing_named_operator_failures(
    solver_case, monkeypatch, scenario
):
    task, _, runner, checkpoints = solver_case
    acceptance = Path(task.pull_checks[0][-1])
    expectations = (
        '    assert process([{}, {}])["primary"] == (0 if case == "pending" else 2)\n'
        if scenario == "trade_cases"
        else '    assert (second() == 2 if case == "pending" else first() == 1)\n'
    )
    acceptance.write_text(
        "import pytest\nfrom fixture_calc import first, second, process\n\n"
        '@pytest.mark.parametrize("case", ["pending", "preserve"])\n'
        "def test_review_cases(case):\n" + expectations
    )
    base = SOURCE_TEXT.replace(FIRST + SECOND, SECOND + FIRST)
    current = SOURCE_TEXT.replace("return -1", "return 1")
    reordered = current.replace(FIRST + SECOND, SECOND + FIRST)
    draft = DraftCheckpoint(
        review_solver._CHECKPOINT,
        task.base_sha,
        review_solver.patch_for({SOURCE: base}, {SOURCE: current}),
        guidance=review_solver._GUIDANCE,
        last_editor=("free-checkpoint", "model-1"),
    )
    check = (
        sys.executable,
        "-c",
        "from fixture_calc import first, process\n"
        + (
            'assert process([{}, {}])["primary"] == 2'
            if scenario == "static_regression"
            else "assert first() == 1"
        ),
    )
    task = replace(task, checkpoints={draft.key: draft}, checks=(*task.checks, check))
    reordered_patch = review_solver.patch_for({SOURCE: base}, {SOURCE: reordered})
    tracked = (
        "import pytest\nfrom fixture_calc import process\n\n"
        "def test_stock():\n    item = {}\n"
        '    assert process([item, item])["primary"] == 0\n'
    )
    if scenario == "renamed_stock":
        tracked += (
            '\nreason = process([{}, {}])["primary"]\n'
            '@pytest.mark.parametrize("value", [1], ids=[f"identity-{reason}"])\n'
            "def test_passing_companion(value):\n    assert value == 1\n"
        )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, source in {SOURCE: base, "tests/test_existing.py": tracked}.items():
            data = source.encode()
            item = tarfile.TarInfo(path)
            item.size = len(data)
            snapshot.addfile(item, BytesIO(data))

    runs = []
    actual = review_solver.evidence

    def evidence(*args, **kwargs):
        checked = actual(*args, **kwargs)
        if args[4] == reordered_patch:
            runs.append({node for _, node in checked})
        return checked

    monkeypatch.setattr(review_solver, "evidence", evidence)

    class Gateway(FreeGateway):
        calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert runs, "the one-shot guard trial must precede model requests"
                prompt = kwargs["messages"][-1]["content"]
                assert "test_review_cases[pending]" in prompt.splitlines()[0]
                checkout = prompt.partition("\nCurrent checkout:\n")[2]
                original_order = checkout.index(FIRST.splitlines()[0]) < checkout.index(
                    SECOND.splitlines()[0]
                )
                assert original_order is (scenario != "accept")
            if scenario != "accept":
                raise ProviderExhausted(None, {"fixture": "not available"})
            return super().complete(**kwargs)

    gateway = Gateway(
        edit(SOURCE, "    return -2\n", "    return 2\n"),
        edit(
            NEW_TEST,
            "",
            "from fixture_calc import second\n\ndef test_second():\n    assert second() == 2\n",
        ),
    )
    executor = PlainExecutor(
        runner,
        gateway,
        Limits(model_requests=2, concurrent_probes=1),
        verification_runner=FixtureRunner(runner.path / "verification"),
    )
    executor.verification_runner.path.mkdir()
    budget = _Budget(executor.limits, lambda: False, task)

    if scenario == "accept":
        result = review_solver.solve(
            executor, task, archive.getvalue(), budget, "fixture-version"
        )
        assert all(check.returncode == 0 for check in result.checks)
        assert b"+    return 1\n" in result.patch and b"+    return 2\n" in result.patch
        assert NEW_TEST.encode() in result.patch
        assert len(gateway.prompts) == 2
        assert len(checkpoints) == 1 and checkpoints[0].patch != reordered_patch
    else:
        with pytest.raises((ProviderExhausted, AttemptRejected)):
            review_solver.solve(
                executor, task, archive.getvalue(), budget, "fixture-version"
            )
        assert not checkpoints
    assert len(runs) == 1
    assert any(
        node.endswith("[preserve]" if scenario == "trade_cases" else "[pending]")
        for node in runs[0]
    )
    if scenario == "static_regression":
        assert "check 3 (exit 1)" in runs[0]
