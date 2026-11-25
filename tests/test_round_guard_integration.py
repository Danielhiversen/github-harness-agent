import tarfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from test_executor import FixtureRunner
from test_incremental_solver import BASE_SOURCE, NEW_TEST, SOURCE, FreeGateway, edit
from test_incremental_solver import solver_case as build_solver_case

from github_harness_agent import review_solver
from github_harness_agent.checkpoint import DraftCheckpoint, SolverRecovery
from github_harness_agent.executor import (
    AttemptRejected,
    Limits,
    PlainExecutor,
    _Budget,
)

solver_case = build_solver_case
SHARED = (
    "from __future__ import annotations\nfrom collections import deque\n\n"
    + BASE_SOURCE
    + """
class Work:
    def __init__(self, depth=0):
        self.depth = depth

    def children(self) -> list[Work]:
        if self.depth >= 2:
            return []
        return [Work(self.depth + 1)]


class Scheduler:
    def seed(self):
        return [Work()]

    def run(self, *, rounds):
"""
)
PINNED = (
    SHARED
    + """        nodes = self.seed()
        visited = []
        for _ in range(rounds):
            next_nodes = []
            for item in nodes:
                visited.append(item.depth)
                next_nodes.extend(item.children())
            nodes = next_nodes
        return visited
"""
)
CURRENT = (
    SHARED.replace("return -1", "return 1").replace("return -2", "return 2")
    + """        if rounds < 1:
            raise ValueError("positive rounds required")
        roots = deque(self.seed())
        queued = deque()
        visited = []
        while roots or queued:
            if roots:
                item = roots.popleft()
            else:
                item = queued.popleft()
            visited.append(item.depth)
            if item.depth < 2:
                next_nodes = item.children()
                for child in next_nodes:
                    queued.append(child)
        return visited
"""
)


@pytest.mark.parametrize(
    "scenario", ("verified", "resume", "after_stock", "stock_regression", "changed_selection")
)
def test_round_limit_trial_stays_private_until_required_new_tests(
    solver_case, monkeypatch, scenario
):
    task, _, runner, checkpoints = solver_case
    acceptance = Path(task.pull_checks[0][-1])
    with acceptance.open("a") as stream:
        stream.write(
            "\nfrom fixture_calc import Scheduler\n\ndef test_one_round():\n"
            "    assert Scheduler().run(rounds=1) == [0]\n"
        )
    stock = "def test_repository():\n    assert True\n"
    baseline, working = PINNED, CURRENT
    health = "\n\ndef repository_health():\n    return False\n"
    stock_fix = edit(SOURCE, "    return False\n", "    return True\n")
    if scenario == "after_stock":
        baseline += health.replace("False", "True")
        working += health
        stock = "from fixture_calc import repository_health\n\ndef test_repository():\n    assert repository_health()\n"
    if scenario == "stock_regression":
        stock = (
            "from fixture_calc import Scheduler\n\ndef test_repository():\n"
            "    assert Scheduler().run(rounds=1) == [0, 1, 2]\n"
        )
    if scenario == "changed_selection":
        stock = (
            "import pytest\nfrom fixture_calc import Scheduler\n\n"
            "@pytest.mark.parametrize('marker', [0], ids=[str(len(Scheduler().run(rounds=1)))])\n"
            "def test_repository(marker):\n    assert marker == 0\n"
        )
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as snapshot:
        for path, value in {SOURCE: baseline, "tests/test_existing.py": stock}.items():
            data = value.encode()
            item = tarfile.TarInfo(path)
            item.size = len(data)
            snapshot.addfile(item, BytesIO(data))

    patch = review_solver.patch_for({SOURCE: baseline}, {SOURCE: working})
    checkpoint = DraftCheckpoint(
        review_solver._CHECKPOINT,
        task.base_sha,
        patch,
        guidance=review_solver._GUIDANCE,
        last_editor=("free-fixture-route", "fixture/model"),
    )
    recovery = []
    task = replace(
        task,
        checkpoints={checkpoint.key: checkpoint},
        save_solver_recovery=lambda state, durable: recovery.append(state) if durable else None,
    )
    actual = review_solver.prepare_round_limit_guard
    prepared = []
    selection_runs = []
    collect = review_solver.stock_collection

    def collect_selection(*args, **kwargs):
        result = collect(*args, **kwargs)
        selection_runs.append((args[6], bool(kwargs.get("failed")), bool(result)))
        return result

    monkeypatch.setattr(review_solver, "stock_collection", collect_selection)

    def prepare(*args, **kwargs):
        guard = actual(*args, **kwargs)
        prepared.append(guard)
        return guard

    monkeypatch.setattr(review_solver, "prepare_round_limit_guard", prepare)

    class Gateway(FreeGateway):
        def complete(self, **request):
            prompt = request["messages"][-1]["content"]
            if scenario == "after_stock" and not self.prompts:
                assert not prepared and not checkpoints
                return super().complete(**request)
            if scenario in {"verified", "resume", "after_stock"}:
                assert prompt.startswith("Selected failure: new regression tests"), selection_runs
                assert len(checkpoints) == 1 + int(scenario == "after_stock")
                assert b"if item.depth + 1 < rounds else []" in checkpoints[-1].patch
                assert NEW_TEST.encode() not in checkpoints[-1].patch
            else:
                assert "test_one_round" in prompt.splitlines()[0]
                assert "Verified draft source observation" in prompt
                assert "self.depth = depth" in prompt and "self.depth + 1" in prompt
                assert "if item.depth + 1 < rounds else []" not in prompt
                assert not checkpoints
            return super().complete(**request)

    model_test = (
        "from fixture_calc import Scheduler, first\n\ndef test_two_rounds_preserve_depth_order():\n"
        "    assert Scheduler().run(rounds=2) == [0, 1]\n    assert first() == 1\n"
    )
    extra_test = "\n\ndef test_reviewed_first():\n    assert first() == 1\n"
    complete_test = model_test + extra_test
    if scenario == "resume":
        test_edit = complete_test.replace("== [0, 1]", "== [1]")
    else:
        test_edit = edit(NEW_TEST, "", model_test)
    replies = (
        (stock_fix, test_edit)
        if scenario == "after_stock"
        else ((test_edit,) if scenario in {"verified", "resume"} else ("invalid edit",))
    )
    gateway = Gateway(*replies)
    verifier = runner.path / "verification"
    verifier.mkdir()
    verification_runner = FixtureRunner(verifier)

    def solve(current_task, current_gateway, model_requests):
        executor = PlainExecutor(
            runner,
            current_gateway,
            Limits(model_requests=model_requests, concurrent_probes=1),
            verification_runner=verification_runner,
        )
        budget = _Budget(executor.limits, lambda: False, current_task)
        return review_solver.solve(executor, current_task, archive.getvalue(), budget, "fixture")

    if scenario == "resume":
        with pytest.raises(AttemptRejected, match="test_two_rounds_preserve_depth_order"):
            solve(task, gateway, len(replies))
        [retained] = recovery
        assert retained.test_seed.passing == ("test_reviewed_first",)
        resumed = replace(
            task,
            checkpoints={checkpoint.key: checkpoints[-1]},
            solver_recovery=SolverRecovery.from_data(retained.to_data()),
        )
        resumed_gateway = Gateway(complete_test)
        result = solve(resumed, resumed_gateway, 1)
        assert "def test_reviewed_first" in resumed_gateway.prompts[0]
        assert "def test_two_rounds_preserve_depth_order" not in resumed_gateway.prompts[0]
        assert b"def test_reviewed_first" in result.patch
    elif scenario in {"verified", "after_stock"}:
        result = solve(task, gateway, len(replies))
    else:
        with pytest.raises(AttemptRejected):
            solve(task, gateway, len(replies))
    if scenario in {"verified", "resume", "after_stock"}:
        assert all(check.returncode == 0 for check in result.checks)
        assert b"if item.depth + 1 < rounds else []" in result.patch
        assert NEW_TEST.encode() in result.patch
    assert len(prepared) == 1 and prepared[0] is not None
    assert len(gateway.prompts) == len(replies)
    assert len(checkpoints) == int(scenario in {"verified", "resume", "after_stock"}) + int(
        scenario == "after_stock"
    )
    if scenario == "stock_regression":
        assert not selection_runs
    else:
        assert len(selection_runs) == 4 and all(run[2] for run in selection_runs)
