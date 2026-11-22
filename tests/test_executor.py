import json
import os
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from threading import Lock
from types import SimpleNamespace

import pytest

from github_harness_agent.executor import (
    Cancelled,
    DockerRunner,
    ExecutionError,
    Harness2Executor,
    LimitExceeded,
    Limits,
    NeedsClarification,
    PlainExecutor,
    TaskSnapshot,
)
from github_harness_agent.providers import ProviderExhausted
from github_harness_agent.worker import CommandOutputExceeded


class FixtureRunner:
    """Only executes the deterministic fixture commands in temporary directories."""

    def __init__(self, path):
        self.path = path
        self.workspaces = []
        self.lock = Lock()

    @contextmanager
    def workspace(self, archive, remaining, cancelled):
        with self.lock:
            root = self.path / str(len(self.workspaces))
            root.mkdir()
            self.workspaces.append(root)
        with tarfile.open(fileobj=BytesIO(archive)) as source:
            source.extractall(root, filter="data")
        yield FixtureWorkspace(root)


class FixtureWorkspace:
    def __init__(self, root):
        self.root = root

    def command(self, argv, timeout, cancelled, *, data=b"", cap=65536):
        result = subprocess.run(
            argv,
            cwd=self.root,
            input=data,
            capture_output=True,
            timeout=timeout,
            env=dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull),
        )
        if len(result.stdout) + len(result.stderr) > cap:
            raise CommandOutputExceeded("Command output exceeded the configured size")
        return result


class ScriptedGateway:
    def __init__(self, code, *, python=sys.executable):
        self.code = code
        self.python = python
        self.roles = []
        self.composition_evidence = {}

    def complete(
        self, *, role, messages, tools=(), timeout=None, exclude=(), after_provider=None
    ):
        self.roles.append(role)
        content = "Done"
        calls = ()
        if role in {"proposal", "composition"}:
            if role == "composition":
                self.composition_evidence = json.loads(messages[-1]["content"])[
                    "evidence"
                ]
            content = json.dumps(
                {"system_prompt": "Inspect the result.", "notes": "Fixture"}
            )
        elif messages[-1]["role"] == "user" and self.code:
            code = self.code.get(role) if isinstance(self.code, dict) else self.code
            calls = (
                {
                    "id": "edit",
                    "type": "function",
                    "function": {
                        "name": "run",
                        "arguments": json.dumps({"argv": [self.python, "-c", code]}),
                    },
                },
            )
            content = None
        return SimpleNamespace(
            content=content,
            tool_calls=calls,
            role=role,
            route="fixture",
            model="deterministic",
            configured_model="fixture-model-id",
            usage={"input_tokens": 1, "output_tokens": 2},
        )


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "number.txt").write_text("3\n")
    (source / "test_number.py").write_text(
        "from pathlib import Path\nassert Path('number.txt').read_text() == '4\\n'\n"
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "base",
        ],
        cwd=source,
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    task = TaskSnapshot(
        "org/fixture",
        sha,
        "Set the number to 4.",
        ((sys.executable, "test_number.py"),),
    )
    return task, source, FixtureRunner(tmp_path)


def test_plain_verifies_patch_in_independent_checkout(fixture):
    task, source, runner = fixture
    gateway = ScriptedGateway("open('number.txt','w').write('4\\n')")
    result = PlainExecutor(runner, gateway).execute(task, source)
    assert b"+4\n" in result.patch
    assert result.checks[0].returncode == 0
    assert result.usage == {"input_tokens": 2, "output_tokens": 4}
    assert len(runner.workspaces) == 2
    assert (source / "number.txt").read_text() == "3\n"


def test_review_test_update_is_verified_with_the_configured_checks(fixture):
    task, source, runner = fixture
    (source / "test_number.py").write_text(
        "from pathlib import Path\n\n"
        "def test_number():\n"
        "    assert Path('number.txt').read_text() == '4\\n'\n\n"
        "def test_neighbor():\n"
        "    assert Path('number.txt').stat().st_size == 2\n"
    )
    subprocess.run(["git", "add", "test_number.py"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "existing tests",
        ],
        cwd=source,
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    task = replace(
        task,
        base_sha=sha,
        checks=((sys.executable, "-m", "pytest", "-q", "-k", "neighbor"),),
    )
    code = (
        "from pathlib import Path; test = Path('test_number.py'); "
        'test.write_text(test.read_text().replace("\'4", "\'3"))'
    )
    with pytest.raises(ExecutionError, match="protected path"):
        PlainExecutor(runner, ScriptedGateway(code)).execute(task, source)
    task = replace(task, test_update="test_number.py::test_number")
    result = PlainExecutor(runner, ScriptedGateway(code)).execute(task, source)
    assert len(result.checks) == 2
    assert all(check.returncode == 0 for check in result.checks)
    assert b"+    assert Path('number.txt').read_text() == '3\\n'" in result.patch

    task = replace(task, require_new_tests=True)
    for empty_module in ("", "; Path('test_new.py').write_text('# TODO: test')"):
        with pytest.raises(ExecutionError, match="new runnable regression tests"):
            PlainExecutor(runner, ScriptedGateway(code + empty_module)).execute(
                task, source
            )

    new_test = "from pathlib import Path\ndef test_regression():\n    assert Path('number.txt').read_text() == '3\\n'\n"
    gateway = ScriptedGateway(code + f"; Path('test_new.py').write_text({new_test!r})")
    result = PlainExecutor(runner, gateway).execute(task, source)
    assert len(result.checks) == 4
    assert result.checks[-1].command[-1] == "test_new.py"
    assert all(check.returncode == 0 for check in result.checks)


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("open('number.txt','w').write('4\\n')", None),
        ("open('number.txt','w').write('5\\n')", "Required check"),
        ("print('inspection only')", "empty patch"),
    ],
)
def test_capped_rollout_keeps_only_a_verified_patch(fixture, code, error):
    task, source, runner = fixture
    path = source.parent / "verification"
    path.mkdir()
    gateway = ScriptedGateway(code)
    executor = PlainExecutor(
        runner,
        gateway,
        Limits(model_requests=1, rollout_requests=1),
        verification_runner=FixtureRunner(path),
    )
    if error:
        with pytest.raises(ExecutionError, match=error):
            executor.execute(task, source)
    else:
        result = executor.execute(task, source)
        assert b"+4\n" in result.patch
        assert result.checks[0].returncode == 0
    assert gateway.roles == ["coding"]


def test_capped_harness2_rollouts_leave_shared_budget_for_refinement(fixture):
    pytest.importorskip("harness2")
    task, source, runner = fixture
    path = source.parent / "verification"
    path.mkdir()
    gateway = ScriptedGateway("open('number.txt','w').write('4\\n')")
    executor = Harness2Executor(
        runner,
        gateway,
        Limits(model_requests=7, rollout_requests=1),
        verification_runner=FixtureRunner(path),
    )
    result = executor.execute(task, source)
    assert result.checks[0].returncode == 0
    assert len(gateway.roles) == 7
    assert gateway.roles.count("proposal") == 2
    assert gateway.roles.count("composition") == 1
    assert all(
        "rollout model-turn limit" in item.trajectory
        for item in result.refinement.candidates
    )


@pytest.mark.parametrize("candidate", [4, 5])
def test_trusted_verification_runner_is_separate_from_model_workspace(
    fixture, candidate
):
    task, source, coding = fixture
    verification_path = source.parent / "verification"
    verification_path.mkdir()
    (verification_path / "trusted.py").write_text(
        "from pathlib import Path\nassert Path('number.txt').read_text() == '4\\n'\n"
    )
    verification = FixtureRunner(verification_path)
    task = replace(task, checks=((sys.executable, "../trusted.py"),))
    gateway = ScriptedGateway(
        "from pathlib import Path\n"
        "assert not Path('../trusted.py').exists()\n"
        f"Path('number.txt').write_text('{candidate}\\n')"
    )
    executor = PlainExecutor(coding, gateway, verification_runner=verification)
    if candidate == 4:
        result = executor.execute(task, source)
        assert result.checks[0].returncode == 0
        assert len(verification.workspaces) == 1
    else:
        with pytest.raises(ExecutionError, match="failed on the baseline"):
            executor.execute(task, source)
        assert len(verification.workspaces) == 2
    assert len(coding.workspaces) == 1


def test_rejects_patch_that_weakens_trusted_check(fixture):
    task, source, runner = fixture
    gateway = ScriptedGateway("open('test_number.py','w').write('pass\\n')")
    with pytest.raises(ExecutionError, match="protected"):
        PlainExecutor(runner, gateway).execute(task, source)


def test_adds_new_regression_without_loading_it_during_trusted_checks(fixture):
    task, source, runner = fixture
    task = replace(
        task,
        checks=(
            (
                sys.executable,
                "-c",
                "import runpy; from pathlib import Path; "
                "[runpy.run_path(str(path)) for path in Path('.').glob('test_*.py')]",
            ),
        ),
    )
    gateway = ScriptedGateway(
        "open('number.txt','w').write('4\\n'); "
        "open('test_regression.py','w').write(\"print('new regression ran')\\n\")"
    )
    result = PlainExecutor(runner, gateway).execute(task, source)
    assert b"test_regression.py" in result.patch
    assert [check.output.strip() for check in result.checks] == [
        "",
        "new regression ran",
    ]
    assert len(runner.workspaces) == 3


@pytest.mark.parametrize(
    "change",
    [
        "open('conftest.py','w').write('pass\\n')",
        "__import__('os').symlink('number.txt', 'test_regression.py')",
        "__import__('os').mkdir('tests'); open('tests/__init__.py','w').write('pass\\n')",
        "__import__('os').unlink('test_number.py'); "
        "__import__('os').symlink('number.txt', 'test_number.py')",
    ],
)
def test_rejects_test_configuration_and_symlink_replacements(fixture, change):
    task, source, runner = fixture
    with pytest.raises(ExecutionError, match="protected|symlink"):
        PlainExecutor(runner, ScriptedGateway(change)).execute(task, source)


def test_new_test_cannot_bypass_existing_checks(fixture):
    task, source, runner = fixture
    task = replace(
        task,
        checks=(
            (
                sys.executable,
                "-c",
                "import runpy; from pathlib import Path; "
                "Path('test_regression.py').exists() or runpy.run_path('test_number.py')",
            ),
        ),
    )
    gateway = ScriptedGateway(
        "open('number.txt','w').write('99\\n'); "
        "open('test_regression.py','w').write('pass\\n')"
    )
    with pytest.raises(ExecutionError, match="Required check"):
        PlainExecutor(runner, gateway).execute(task, source)


def test_model_can_continue_after_an_oversized_tool_read(fixture):
    task, source, runner = fixture

    class NarrowingGateway(ScriptedGateway):
        def complete(self, **kwargs):
            if kwargs["messages"][-1]["role"] == "user":
                return super().complete(**kwargs)
            if "output limit" in kwargs["messages"][-1]["content"]:
                self.code = "open('number.txt','w').write('4\\n')"
                return super().complete(**(kwargs | {"messages": [{"role": "user"}]}))
            return super().complete(**kwargs)

    result = PlainExecutor(runner, NarrowingGateway("print('x' * 80000)")).execute(
        task, source
    )
    assert b"+4\n" in result.patch


def test_oversized_checkout_reports_the_archive_limit(fixture):
    task, source, runner = fixture
    executor = PlainExecutor(runner, ScriptedGateway(None), Limits(archive_bytes=1024))
    with pytest.raises(LimitExceeded, match="archive limit"):
        executor.execute(task, source)
    assert runner.workspaces == []


def test_rejects_empty_patch(fixture):
    task, source, runner = fixture
    with pytest.raises(ExecutionError, match="empty patch"):
        PlainExecutor(runner, ScriptedGateway(None)).execute(task, source)


def test_cancellation_does_not_allocate_worker(fixture):
    task, source, runner = fixture
    gateway = ScriptedGateway(None)
    with pytest.raises(Cancelled):
        PlainExecutor(runner, gateway).execute(task, source, cancelled=lambda: True)
    assert runner.workspaces == []
    assert gateway.roles == []


def test_model_request_quota_applies_after_tool_calls(fixture):
    task, source, runner = fixture
    gateway = ScriptedGateway("open('number.txt','w').write('4\\n')")
    with pytest.raises(LimitExceeded, match="model request"):
        PlainExecutor(runner, gateway, Limits(model_requests=1)).execute(task, source)
    assert gateway.roles == ["coding"]


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (None, "empty patch"),
        ("open('test_number.py','w').write('pass\\n')", "protected"),
    ],
)
def test_candidate_failure_identifies_actual_configured_model(fixture, code, reason):
    from github_harness_agent.executor import AttemptRejected

    class RoutedGateway(ScriptedGateway):
        def complete(self, *, exclude=(), provider_offset=0, **kwargs):
            assert exclude == (("other", "different"),)
            assert provider_offset == 2
            return super().complete(**kwargs)

    task, source, runner = fixture
    task = replace(task, excluded_models=(("other", "different"),), provider_offset=2)
    with pytest.raises(AttemptRejected, match=reason) as error:
        PlainExecutor(runner, RoutedGateway(code)).execute(task, source)
    assert error.value.candidate.route == "fixture"
    assert error.value.candidate.model == "deterministic"
    assert error.value.candidate.configured_model == "fixture-model-id"


def test_model_budget_marks_only_the_consuming_candidate(fixture):
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    gateway = ScriptedGateway("open('number.txt','w').write('4\\n')")
    with pytest.raises(AttemptRejected, match="model request") as error:
        PlainExecutor(runner, gateway, Limits(model_requests=1)).execute(task, source)
    assert error.value.candidate.configured_model == "fixture-model-id"


def test_command_budget_rejects_the_model_asking_for_extra_tools(fixture):
    from github_harness_agent.executor import AttemptRejected

    class TwoCommands(ScriptedGateway):
        def complete(self, **kwargs):
            reply = super().complete(**kwargs)
            reply.tool_calls += (dict(reply.tool_calls[0], id="extra"),)
            return reply

    task, source, runner = fixture
    gateway = TwoCommands("open('number.txt','w').write('4\\n')")
    with pytest.raises(AttemptRejected, match="command budget") as error:
        PlainExecutor(runner, gateway, Limits(commands=1)).execute(task, source)
    assert error.value.candidate.configured_model == "fixture-model-id"


class EditGateway(ScriptedGateway):
    def __init__(self, edits):
        super().__init__(None)
        self.edits = iter(edits)
        self.observations = []

    def complete(self, *, messages, tools, **kwargs):
        reply = super().complete(messages=messages, tools=tools, **kwargs)
        assert "edit_file" in {tool["function"]["name"] for tool in tools}
        if messages[-1]["role"] == "tool":
            self.observations.append(messages[-1]["content"])
        if edit := next(self.edits, None):
            reply.tool_calls = (
                {
                    "id": f"edit-{len(self.roles)}",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(edit),
                    },
                },
            )
        return reply


def test_exact_edit_retries_then_checkpoints_and_verifies(fixture):
    task, source, runner = fixture
    receipts = []
    gateway = EditGateway(
        [
            {"path": "../source/number.txt", "old_text": "3", "new_text": "4"},
            {"path": "number.txt", "old_text": "missing", "new_text": "4"},
            {"path": "number.txt", "old_text": "3", "new_text": "4"},
            {
                "path": "test_regression.py",
                "new_text": "from pathlib import Path\n\n\ndef test_new():\n"
                "    assert Path('number.txt').read_text() == '4\\n'\n",
            },
        ],
    )
    result = PlainExecutor(runner, gateway).execute(
        replace(task, save_checkpoint=receipts.append), source
    )
    assert b"-3\n+4\n" in result.patch
    assert b"test_regression.py" in result.patch
    assert receipts[-1].patch == result.patch
    assert receipts[-1].last_editor == ("fixture", "fixture-model-id")
    assert len(receipts) == 2
    assert "relative checkout file path" in gateway.observations[0]
    assert "old_text must occur exactly once" in gateway.observations[1]
    assert (source / "number.txt").read_text() == "3\n"
    assert not (source / "test_regression.py").exists()
    assert sorted(path.name for path in runner.workspaces[0].glob(".gha-edit-*")) == []


def test_exact_edit_preserves_protected_path_checks(fixture):
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    receipts = []
    gateway = EditGateway(
        [{"path": "test_number.py", "old_text": "== '4", "new_text": "== '3"}]
    )
    with pytest.raises(AttemptRejected, match="protected path: test_number.py"):
        PlainExecutor(runner, gateway).execute(
            replace(task, save_checkpoint=receipts.append), source
        )
    assert receipts == []


def test_exact_edit_refuses_to_follow_symlink_parents(tmp_path):
    from github_harness_agent.file_edit import EDIT_COMMAND

    root = tmp_path / "checkout"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    file = outside / "target.txt"
    file.write_text("initial")
    (root / "link").symlink_to(outside, target_is_directory=True)
    result = FixtureWorkspace(root).command(
        EDIT_COMMAND,
        10,
        lambda: False,
        data=json.dumps(
            {"path": "link/target.txt", "old_text": "initial", "new_text": "updated"}
        ).encode(),
    )
    assert result.returncode != 0
    assert file.read_text() == "initial"


def test_check_classification_preserves_explicit_tool_edits():
    from github_harness_agent.check_artifacts import check_label

    assert check_label(("uv", "run", "ruff", "format", "--check"), ()) == "ruff"
    assert check_label(("uv", "run", "ruff", "format"), ()) is None
    assert check_label(("ruff", "check", "--fix"), ()) is None
    assert check_label(("jest", "--updateSnapshot"), ()) is None


def test_rollout_stages_model_edits_without_check_byproducts(fixture):
    task, source, runner = fixture
    (source / "tracked runtime.dat").write_text("previous runtime output")
    (source / "overwritten.dat").write_bytes(b"\x00baseline")
    (source / "claimed-tracked.txt").write_text("baseline")
    (source / "check_suite.py").write_text(
        "from pathlib import Path\n"
        "from subprocess import run\n"
        "Path('generated [1].dat').write_bytes(b'\\x00byproduct')\n"
        "Path('tracked runtime.dat').write_text('regenerated')\n"
        "Path('overwritten.dat').write_bytes(b'\\x00rewrite')\n"
        "Path('number.txt').write_text('rewrite')\n"
        "Path('claimed-tracked.txt').write_text('generated')\n"
        "Path('keep.txt').write_text('rewrite')\n"
        "if not Path('claim.txt').exists():\n"
        "    Path('claim.txt').write_text('generated')\n"
        "run(['git', 'add', 'generated [1].dat', 'overwritten.dat', 'number.txt'], check=True)\n"
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=f@example.test",
            "commit",
            "-qm",
            "include check fixture",
        ],
        cwd=source,
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    receipts = []
    task = replace(
        task,
        base_sha=sha,
        checks=(
            (
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "assert not Path('generated [1].dat').exists(); "
                "assert not Path('tracked runtime.dat').exists(); "
                "assert Path('overwritten.dat').read_bytes() == b'\\x00baseline'; "
                "assert Path('keep.txt').read_text() == 'intentional before the check'; "
                "assert Path('claimed-tracked.txt').read_text() == 'intentional tracked edit'; "
                "assert Path('number.txt').read_text() == '4\\n'",
            ),
        ),
        save_checkpoint=receipts.append,
    )
    actions = [
        ("edit_file", {"path": "number.txt", "old_text": "3", "new_text": "4"}),
        ("edit_file", {"path": "keep.txt", "new_text": "intentional before the check"}),
        ("run", {"argv": ["rm", "tracked runtime.dat"]}),
        ("run", {"argv": [sys.executable, "-m", "unittest", "check_suite"]}),
        (
            "edit_file",
            {
                "path": "claim.txt",
                "old_text": "generated",
                "new_text": "intentional after the check",
            },
        ),
        (
            "edit_file",
            {
                "path": "claimed-tracked.txt",
                "old_text": "generated",
                "new_text": "intentional tracked edit",
            },
        ),
    ]

    class CheckGateway(ScriptedGateway):
        def complete(self, **kwargs):
            reply = super().complete(**kwargs)
            turn = len(self.roles)
            if turn <= len(actions):
                name, arguments = actions[turn - 1]
                reply.tool_calls = (
                    {
                        "id": f"check-{turn}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    },
                )
            return reply

    result = PlainExecutor(runner, CheckGateway(None)).execute(task, source)
    assert b"generated [1].dat" not in result.patch
    assert b"overwritten.dat" not in result.patch
    assert b"deleted file mode" in result.patch
    assert b"tracked runtime.dat" in result.patch
    assert b"intentional before the check" in result.patch
    assert b"intentional after the check" in result.patch
    assert receipts[-1].patch == result.patch
    assert all(b"generated [1].dat" not in draft.patch for draft in receipts)
    assert (runner.workspaces[0] / "generated [1].dat").exists()
    assert (runner.workspaces[0] / "tracked runtime.dat").exists()

    draft = receipts[-1]
    actions[:] = [actions[3]]
    resumed = PlainExecutor(runner, CheckGateway(None)).execute(
        replace(task, checkpoints={draft.key: draft}), source
    )
    assert resumed.patch == result.patch


def with_resumed_number_edit(task):
    from github_harness_agent.checkpoint import DraftCheckpoint

    draft = DraftCheckpoint(
        "plain/coding",
        task.base_sha,
        b"diff --git a/number.txt b/number.txt\n"
        b"--- a/number.txt\n+++ b/number.txt\n@@ -1 +1 @@\n-3\n+4\n",
    )
    return replace(task, checkpoints={draft.key: draft})


@pytest.mark.parametrize(
    ("limit", "threads", "edit_turn", "resumed", "model_limit", "expected_turns"),
    [
        (None, 4, None, False, 40, 12),
        (None, 1, 12, False, 40, 12),
        (None, 4, 12, False, 40, 20),
        (None, 4, 12, False, 14, 14),
        (None, 4, None, True, 40, 12),
        (2, 4, 2, False, 40, 2),
    ],
)
def test_review_rollout_bounds_context_reads_and_stages_last_edit(
    fixture, limit, threads, edit_turn, resumed, model_limit, expected_turns
):
    from github_harness_agent.executor import AttemptRejected
    from github_harness_agent.task_context import TaskContext

    class ReviewingGateway(ScriptedGateway):
        def __init__(self):
            super().__init__(None)
            self.first_reminder_turn = None

        def complete(self, *, messages, **kwargs):
            reply = super().complete(messages=messages, **kwargs)
            turn = len(self.roles)
            if (
                self.first_reminder_turn is None
                and "still has no draft patch" in messages[0]["content"]
            ):
                self.first_reminder_turn = turn
            name = "read_task_context"
            arguments = {"section": "sections", "page": 1}
            if turn == edit_turn:
                name = "edit_file"
                arguments = {
                    "path": "number.txt",
                    "old_text": "3",
                    "new_text": "4",
                }
            reply.tool_calls = (
                {
                    "id": f"review-{turn}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                },
            )
            return reply

    task, source, runner = fixture
    task = replace(
        task,
        verify_base_first=True,
        inline_review_threads=threads,
        context=TaskContext(task.instructions, {"sections": "Review details"}, {}),
    )
    if resumed:
        task = with_resumed_number_edit(task)
    gateway = ReviewingGateway()
    executor = PlainExecutor(
        runner,
        gateway,
        Limits(model_requests=model_limit, rollout_requests=limit),
        verification_runner=runner,
    )
    if edit_turn or resumed:
        assert b"-3\n+4\n" in executor.execute(task, source).patch
    else:
        with pytest.raises(AttemptRejected, match="empty patch"):
            executor.execute(task, source)
    assert len(gateway.roles) == expected_turns
    assert gateway.first_reminder_turn == (4 if limit is None and not resumed else None)


@pytest.mark.parametrize(
    ("limit", "model_limit", "expected_calls"),
    [(None, 40, 2), (None, 1, 1), (4, 40, 1)],
)
def test_empty_review_prose_reminds_once_without_spending_reserved_budget(
    fixture, limit, model_limit, expected_calls
):
    from github_harness_agent.executor import AttemptRejected

    class ProseGateway(ScriptedGateway):
        def complete(self, *, messages, **kwargs):
            reminder = "The checkout still has no patch"
            assert messages[0]["content"].count(reminder) == bool(self.roles)
            return super().complete(messages=messages, **kwargs)

    task, source, runner = fixture
    gateway = ProseGateway(None)
    executor = PlainExecutor(
        runner,
        gateway,
        Limits(model_requests=model_limit, rollout_requests=limit),
        verification_runner=runner,
    )
    with pytest.raises(AttemptRejected, match="empty patch"):
        executor.execute(replace(task, verify_base_first=True), source)
    assert len(gateway.roles) == expected_calls


@pytest.mark.parametrize(
    ("review", "limit", "model_limit", "create_test", "focused_turns", "calls"),
    [
        (True, None, 40, True, 1, 3),
        (True, None, 40, False, 1, 3),
        (True, None, 2, True, 1, 2),
        (True, None, 1, True, 0, 1),
        (True, 4, 40, True, 0, 1),
        (False, None, 40, True, 0, 1),
    ],
)
def test_review_finish_can_focus_once_on_missing_regression_tests(
    fixture, review, limit, model_limit, create_test, focused_turns, calls
):
    from github_harness_agent.executor import AttemptRejected

    class RegressionGateway(ScriptedGateway):
        def __init__(self):
            super().__init__(None)
            self.focused = 0

        def complete(self, *, messages, tools, **kwargs):
            reply = super().complete(messages=messages, tools=tools, **kwargs)
            names = [tool["function"]["name"] for tool in tools]
            if names != ["edit_file"]:
                assert names == ["run", "edit_file"]
                assert "This turn is for" not in messages[0]["content"]
                return reply
            self.focused += 1
            assert (
                "create a runnable new test file with edit_file"
                in messages[0]["content"]
            )
            if create_test:
                reply.tool_calls = (
                    {
                        "id": "regression",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps(
                                {
                                    "path": "test_regression.py",
                                    "new_text": "from pathlib import Path\n\n\n"
                                    "def test_number_file():\n"
                                    "    assert Path('number.txt').read_text() == '4\\n'\n",
                                }
                            ),
                        },
                    },
                )
            return reply

    task, source, runner = fixture
    task = with_resumed_number_edit(
        replace(task, verify_base_first=review, require_new_tests=True)
    )
    gateway = RegressionGateway()
    executor = PlainExecutor(
        runner,
        gateway,
        Limits(model_requests=model_limit, rollout_requests=limit),
        verification_runner=runner,
    )
    if create_test and focused_turns:
        assert b"test_regression.py" in executor.execute(task, source).patch
    else:
        with pytest.raises(AttemptRejected, match="Add new runnable regression tests"):
            executor.execute(task, source)
    assert gateway.focused == focused_turns
    assert len(gateway.roles) == calls


def test_invalid_tool_output_rejects_the_candidate_before_running_it(fixture):
    from github_harness_agent.executor import AttemptRejected

    class MalformedGateway(ScriptedGateway):
        def complete(self, **kwargs):
            reply = super().complete(**kwargs)
            reply.tool_calls[0]["function"]["arguments"] = "not JSON"
            return reply

    task, source, runner = fixture
    gateway = MalformedGateway("open('number.txt','w').write('4\\n')")
    with pytest.raises(AttemptRejected, match="invalid run tool call") as error:
        PlainExecutor(runner, gateway).execute(task, source)
    assert error.value.candidate.configured_model == "fixture-model-id"
    assert (runner.workspaces[0] / "number.txt").read_text() == "3\n"


def test_baseline_test_failure_does_not_exclude_model(fixture):
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    gateway = ScriptedGateway("open('number.txt','w').write('5\\n')")
    with pytest.raises(ExecutionError, match="baseline") as error:
        PlainExecutor(runner, gateway).execute(task, source)
    assert not isinstance(error.value, AttemptRejected)
    assert len(runner.workspaces) == 3


def test_pull_acceptance_failure_can_choose_another_model_on_a_red_baseline(fixture):
    from dataclasses import replace
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    gateway = ScriptedGateway("open('number.txt','w').write('5\\n')")
    with pytest.raises(AttemptRejected, match="Operator acceptance check 1 exited"):
        PlainExecutor(runner, gateway).execute(
            replace(task, pull_checks=task.checks), source
        )
    assert len(runner.workspaces) == 2


def test_missing_review_test_reports_independent_behavior_failure(fixture):
    task, source, runner = fixture
    acceptance = source.parent / "oracle" / "test_case.py"
    acceptance.parent.mkdir()
    acceptance.write_text(
        "from pathlib import Path\n"
        "def test_boundary():\n"
        "    assert Path('number.txt').read_text() == '4\\n'\n"
    )
    check = (sys.executable, "-m", "pytest", "-q", str(acceptance))
    task = replace(task, require_new_tests=True, pull_checks=(check,))
    with pytest.raises(
        ExecutionError,
        match="new runnable regression tests.*test_case.py::test_boundary",
    ):
        PlainExecutor(
            runner, ScriptedGateway("open('number.txt','w').write('5\\n')")
        ).execute(task, source)


def test_coding_workspace_git_status_is_clean_and_head_is_diffable(fixture):
    task, source, runner = fixture
    code = (
        "import subprocess; from pathlib import Path; "
        "assert not subprocess.check_output(['git','status','--porcelain']); "
        "Path('number.txt').write_text('4\\n'); "
        "assert subprocess.check_output(['git','diff','--name-only','HEAD']).strip() == b'number.txt'"
    )
    result = PlainExecutor(runner, ScriptedGateway(code)).execute(task, source)
    assert b"+4" in result.patch


@pytest.mark.parametrize("coverage", ("optional", "missing", "present"))
def test_review_verification_reports_all_required_check_outcomes(fixture, coverage):
    from github_harness_agent.check_diagnostics import check_progress
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    checks = tuple(
        (sys.executable, "-c", command)
        for command in (
            "raise SystemExit(1)",
            "raise SystemExit(2)",
            "pass",
        )
    )
    task = replace(
        task,
        checks=checks[:1],
        pull_checks=checks[1:],
        verify_base_first=True,
        require_new_tests=coverage != "optional",
    )
    code = "open('number.txt','w').write('5\\n')"
    if coverage == "present":
        code += "; open('test_new.py','w').write('def test_behavior():\\n    assert 1 == 1\\n')"
    gateway = ScriptedGateway(code)
    with pytest.raises(ExecutionError) as error:
        PlainExecutor(runner, gateway).execute(task, source)

    failure = (
        error.value.__cause__
        if isinstance(error.value, AttemptRejected)
        else error.value
    )
    assert tuple(check.returncode for check in failure.checks) == (1, 2, 0)
    assert check_progress(task, failure) == (1, coverage == "present")
    assert "Required check 1 exited with status 1." in str(failure)
    assert "Operator acceptance check 1 exited with status 2." in str(failure)


def test_review_verification_stops_at_infrastructure_failure(fixture):
    from github_harness_agent.check_diagnostics import check_progress
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    checks = tuple(
        (sys.executable, "-c", code)
        for code in ("raise SystemExit(1)", "raise SystemExit(125)", "pass")
    )
    task = replace(task, checks=checks, verify_base_first=True)
    with pytest.raises(ExecutionError) as error:
        PlainExecutor(
            runner, ScriptedGateway("open('number.txt','w').write('5\\n')")
        ).execute(task, source)

    assert not isinstance(error.value, AttemptRejected)
    assert tuple(check.returncode for check in error.value.checks) == (1, 125)
    assert check_progress(task, error.value) is None


def test_patch_induced_check_failure_can_choose_another_model(fixture):
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    task = replace(
        task,
        checks=(
            (
                sys.executable,
                "-c",
                "assert open('number.txt').read().strip() != '5'",
            ),
        ),
    )
    gateway = ScriptedGateway("open('number.txt','w').write('5\\n')")
    with pytest.raises(AttemptRejected, match="Required check 1 exited") as error:
        PlainExecutor(runner, gateway).execute(task, source)
    assert error.value.candidate.configured_model == "fixture-model-id"


def test_clarification_propagates_without_verifying_an_empty_patch(fixture):
    class ClarifyingGateway(ScriptedGateway):
        def complete(self, **kwargs):
            reply = super().complete(**kwargs)
            reply.content = json.dumps(
                {"clarification": "Which number should be written?"}
            )
            return reply

    task, source, runner = fixture
    with pytest.raises(NeedsClarification, match="Which number"):
        PlainExecutor(runner, ClarifyingGateway(None)).execute(task, source)


def test_harness2_sequential_recursion_uses_distinct_workspaces(fixture):
    pytest.importorskip("harness2")
    task, source, runner = fixture
    gateway = ScriptedGateway(
        "from pathlib import Path\n"
        "p=Path('number.txt')\nassert p.read_text() == '3\\n'\np.write_text('4\\n')"
    )
    result = Harness2Executor(runner, gateway).execute(task, source)
    assert len(runner.workspaces) == 5
    assert gateway.roles.count("proposal") == 2
    assert gateway.roles.count("composition") == 1
    assert {evidence.label for evidence in result.refinement.candidates} == {
        "base",
        "inv1",
        "dom1",
        "c1",
    }
    assert "[ACTION]" in gateway.composition_evidence["pool/inv1/rollout.md"]
    assert result.refinement.harness_guidance["c1"] == "Inspect the result."
    assert b"+4\n" in result.patch


def test_harness2_attributes_empty_patch_to_final_rollout_model(fixture):
    pytest.importorskip("harness2")
    from github_harness_agent.executor import AttemptRejected

    class RoutedGateway(ScriptedGateway):
        finished_base = False

        def complete(self, **kwargs):
            if kwargs["role"] == "coding" and self.finished_base:
                self.code = None
                reply = super().complete(**kwargs)
                reply.configured_model = "final-model"
                reply.model = "actual-final-model"
                return reply
            reply = super().complete(**kwargs)
            if kwargs["role"] == "coding" and kwargs["messages"][-1]["role"] == "tool":
                self.finished_base = True
            return reply

    task, source, runner = fixture
    gateway = RoutedGateway("open('number.txt','w').write('4\\n')")
    with pytest.raises(AttemptRejected, match="empty patch") as error:
        Harness2Executor(runner, gateway).execute(task, source)
    assert error.value.candidate.configured_model == "final-model"
    assert error.value.candidate.model == "actual-final-model"


def test_harness2_records_one_proposal_outage_and_uses_available_routes(fixture):
    pytest.importorskip("harness2")

    class LimitedGateway(ScriptedGateway):
        def complete(self, **kwargs):
            if kwargs["role"] == "proposal" and "proposal" not in self.roles:
                self.roles.append("proposal")
                raise ProviderExhausted(time.time() + 60, {"fixture": "rate limited"})
            return super().complete(**kwargs)

    task, source, runner = fixture
    gateway = LimitedGateway("open('number.txt','w').write('4\\n')")
    result = Harness2Executor(runner, gateway).execute(task, source)
    assert result.checks[0].returncode == 0
    assert any(stage["execution_failed"] for stage in result.refinement.stages)


def test_harness2_excludes_probe_that_edits_required_test(fixture):
    pytest.importorskip("harness2")
    task, source, runner = fixture
    gateway = ScriptedGateway(
        {
            "coding": "open('number.txt','w').write('4\\n')",
            "probe": "open('test_number.py','w').write('pass\\n')",
        }
    )
    result = Harness2Executor(runner, gateway).execute(task, source)
    for candidate in result.refinement.candidates:
        if candidate.label in {"inv1", "dom1"}:
            assert candidate.patch_digest is None
            assert "protected path" in candidate.error


def test_linux_worker_executes_without_controller_credentials(fixture, monkeypatch):
    image = os.environ.get("GHA_DOCKER_TEST_IMAGE")
    if not image:
        pytest.skip("Set GHA_DOCKER_TEST_IMAGE to the locally built fixture image.")
    monkeypatch.setenv("GH_TOKEN", "controller-fixture-token")
    task, source, _ = fixture
    task = replace(task, checks=(("python", "test_number.py"),))
    gateway = ScriptedGateway(
        "import os\nassert 'GH_TOKEN' not in os.environ\n"
        "assert os.stat('.').st_uid == os.getuid()\n"
        "assert not os.path.exists('/var/run/docker.sock')\n"
        "open('number.txt','w').write('4\\n')",
        python="python",
    )
    result = PlainExecutor(DockerRunner(image), gateway, Limits(wall_time=45)).execute(
        task, source
    )
    assert b"+4\n" in result.patch


@pytest.mark.parametrize("executor_class", [PlainExecutor, Harness2Executor])
def test_provider_loss_resumes_saved_edit_in_new_worker(fixture, executor_class):
    from github_harness_agent.checkpoint import DraftCheckpoint
    from github_harness_agent.store import LeaseLost, Store

    task, source, runner = fixture
    now = [100]
    database = source.parent / "checkpoint.db"
    store = Store(database, clock=lambda: now[0])
    store.enqueue(task.repository, 8, {"drafts": {}})
    old_lease = job = store.claim(ttl=10)
    save_lock = Lock()

    def persist(draft):
        with save_lock:
            job["data"]["drafts"][draft.key] = draft.to_data()
            store.save(job)

    class DisconnectAfterTool(ScriptedGateway):
        def complete(self, **kwargs):
            if kwargs["messages"][-1]["role"] == "tool":
                raise ProviderExhausted(None)
            return super().complete(**kwargs)

    task = replace(task, save_checkpoint=persist)
    with pytest.raises(ProviderExhausted):
        executor_class(
            runner,
            DisconnectAfterTool(
                "open('number.txt','w').write('4\\n'); "
                "open('notes.txt','w').write('New notes\\n')"
            ),
        ).execute(task, source)
    now[0] += 11
    store = Store(database, clock=lambda: now[0])
    job = store.claim()
    assert job["token"] != old_lease["token"]
    with pytest.raises(LeaseLost):
        store.save(old_lease)
    saved = {
        key: DraftCheckpoint.from_data(value)
        for key, value in job["data"]["drafts"].items()
    }
    key = "plain/coding" if executor_class is PlainExecutor else "harness2/base"
    assert b"notes.txt" in saved[key].patch
    assert saved[key].progress == ("python exited 0",)
    assert saved[key].last_editor == ("fixture", "fixture-model-id")

    resumed = []
    preferred_after = []

    class FinishWithoutRepeating(ScriptedGateway):
        def complete(self, **kwargs):
            if kwargs.get("after_provider"):
                preferred_after.append(kwargs["after_provider"])
            if (
                kwargs["role"] in {"coding", "probe"}
                and "Resuming a saved draft" in kwargs["messages"][1]["content"]
            ):
                if kwargs["messages"][-1]["role"] == "tool":
                    assert kwargs["messages"][-1]["content"].startswith("exit 0")
                reply = super().complete(**kwargs)
                reply.route = "replacement"
                reply.configured_model = "replacement-model"
                if reply.tool_calls:
                    resumed.append(True)
                    extra_file = (
                        "Path('extra.txt').write_text('Replacement standalone edit\\n')"
                        if kwargs.get("after_provider")
                        else "assert Path('extra.txt').read_text() == 'Replacement standalone edit\\n'"
                    )
                    reply.tool_calls[0]["function"]["arguments"] = json.dumps(
                        {
                            "argv": [
                                sys.executable,
                                "-c",
                                "import subprocess; from pathlib import Path; "
                                "assert Path('number.txt').read_text() == '4\\n'; "
                                "assert Path('notes.txt').read_text() == 'New notes\\n'; "
                                "assert b'+4' in subprocess.check_output(['git', 'diff', '--', 'number.txt']); "
                                + extra_file,
                            ],
                        }
                    )
                    if not kwargs.get("after_provider"):
                        reply.tool_calls += (
                            {
                                "id": "check",
                                "type": "function",
                                "function": {
                                    "name": "run",
                                    "arguments": json.dumps(
                                        {
                                            "argv": [
                                                sys.executable,
                                                "-m",
                                                "pytest",
                                                "--version",
                                            ]
                                        }
                                    ),
                                },
                            },
                        )
                return reply
            return super().complete(**kwargs)

    result = executor_class(
        runner,
        FinishWithoutRepeating(
            "open('number.txt','w').write('4\\n'); "
            "open('notes.txt','w').write('Replacement value\\n')"
        ),
    ).execute(replace(task, checkpoints=saved.copy()), source)
    assert len(resumed) == (1 if executor_class is PlainExecutor else 4)
    assert preferred_after == ["fixture"]
    assert b"+New notes" in result.patch
    assert b"+Replacement standalone edit" in result.patch
    final_key = "plain/coding" if executor_class is PlainExecutor else "harness2/c1"
    assert DraftCheckpoint.from_data(job["data"]["drafts"][final_key]).last_editor == (
        "replacement",
        "replacement-model",
    )
    assert result.checks[0].returncode == 0
    assert (source / "number.txt").read_text() == "3\n"


@pytest.mark.parametrize(
    ("resumed", "failure", "baseline_passed"),
    [(True, 1, False), (True, 1, True), (False, 1, False), (True, 126, False)],
)
def test_harness2_falls_back_to_completed_resumed_base_only_on_check_rejection(
    fixture,
    resumed,
    failure,
    baseline_passed,
):
    from github_harness_agent.checkpoint import DraftCheckpoint
    from github_harness_agent.harness2_adapter import BASE_GUIDANCE
    from github_harness_agent.rebase import git

    task, source, runner = fixture
    saved = {}
    if resumed:
        (source / "number.txt").write_text("4\n")
        git(source, "add", "number.txt")
        patch = git(
            source, "diff", "--cached", "--binary", "--full-index", task.base_sha
        )
        git(source, "reset", "--hard", task.base_sha)
        draft = DraftCheckpoint(
            "harness2/base", task.base_sha, patch, guidance=BASE_GUIDANCE
        )
        saved[draft.key] = draft
    accepted = ("3\n", "4\n") if baseline_passed else ("4\n",)
    task = replace(
        task,
        checkpoints=saved,
        checks=(
            (
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    f"sys.exit(0 if Path('number.txt').read_text() in {accepted!r} else {failure})"
                ),
            ),
        ),
    )

    class FinalIntroducesRegression:
        def complete(self, **kwargs):
            selected = (
                kwargs["role"] == "coding"
                and "Inspect the result." in kwargs["messages"][0]["content"]
            )
            value = 99 if selected else 4
            code = f"open('number.txt', 'w').write('{value}\\n')"
            return ScriptedGateway(code).complete(**kwargs)

    executor = Harness2Executor(runner, FinalIntroducesRegression())
    if not resumed or failure == 126:
        with pytest.raises(ExecutionError, match="Required check"):
            executor.execute(task, source)
        return

    result = executor.execute(task, source)
    assert b"+4\n" in result.patch
    assert b"+99\n" not in result.patch
    assert result.checks[0].command == task.checks[0]
    assert result.checks[0].returncode == 0
    assert result.refinement.selected_candidate == "base"
    evidence = {
        candidate.label: candidate for candidate in result.refinement.candidates
    }
    assert evidence["base"].patch_digest == result.patch_digest
    assert "Required check" in evidence["c1"].error
    assert result.route_trace


def test_rollout_reads_omitted_evidence_and_limits_command_output(fixture):
    from github_harness_agent.task_context import build_task_context

    task, source, runner = fixture
    raw = "x" * 3600 + "The missing input is four." + "y" * 3600
    context = build_task_context(raw)
    assert "The missing input is four." not in context.instructions
    task = replace(task, instructions=context.instructions, context=context)
    calls = []

    class Gateway:
        def complete(self, *, role, messages, tools, **kwargs):
            calls.append(messages[-1])
            assert "read_task_context" in {tool["function"]["name"] for tool in tools}
            match len(calls):
                case 1:
                    action = "read_task_context"
                    arguments = {"section": "task", "page": 3}
                case 2:
                    assert json.loads(messages[-1]["content"])["content"].startswith(
                        "The missing input is four."
                    )
                    action = "run"
                    arguments = {
                        "argv": [
                            sys.executable,
                            "-c",
                            (
                                "open('number.txt','w').write('4\\n'); print('head' + 'x' * 10000 + 'tail')"
                            ),
                        ]
                    }
                case _:
                    output = messages[-1]["content"]
                    assert len(output.encode()) <= 6000
                    assert output.startswith("exit 0\nhead") and output.endswith(
                        "tail\n"
                    )
                    return SimpleNamespace(
                        content="Done",
                        tool_calls=(),
                        route="fixture",
                        model="model",
                        configured_model="model",
                        role=role,
                        usage={},
                    )
            tool = {
                "id": str(len(calls)),
                "type": "function",
                "function": {"name": action, "arguments": json.dumps(arguments)},
            }
            return SimpleNamespace(
                content=None,
                tool_calls=(tool,),
                route="fixture",
                model="model",
                configured_model="model",
                role=role,
                usage={},
            )

    result = PlainExecutor(runner, Gateway(), Limits(commands=1)).execute(task, source)
    assert b"+4\n" in result.patch
    assert result.checks[0].returncode == 0


def test_harness2_review_accepts_verified_base_without_proposals(fixture):
    task, source, runner = fixture
    verifier = FixtureRunner(source)
    gateway = ScriptedGateway("open('number.txt','w').write('4\\n')")
    result = Harness2Executor(
        runner, gateway, Limits(model_requests=2), verification_runner=verifier
    ).execute(replace(task, verify_base_first=True), source)

    assert b"+4\n" in result.patch
    assert len(runner.workspaces) == len(verifier.workspaces) == 1
    assert gateway.roles == ["coding", "coding"]
    assert [route.role for route in result.route_trace] == gateway.roles
    assert result.refinement.selected_candidate == "base"
    assert result.refinement.candidates[0].patch_digest == result.patch_digest
    assert result.refinement.stages == ()


def test_harness2_review_retries_failed_base_with_safe_check_summary(fixture):
    from github_harness_agent.rebase import git

    task, source, runner = fixture
    (source / "number.py").write_text(
        "def current_number():\n"
        "    # Baseline source\n"
        "    from pathlib import Path\n"
        "    value = Path('number.txt').read_text()\n"
        "    assert value == '4\\n'\n"
        "    return value\n"
    )
    (source / "test_number.py").write_text(
        "from number import current_number\nimport pytest\n\n"
        "@pytest.mark.parametrize('expected', [pytest.param('4\\n', id='private-param')])\n"
        "def test_number(expected):\n"
        "    assert current_number() == expected\n"
    )
    git(source, "add", "test_number.py", "number.py")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "test fixture",
    )
    base = git(source, "rev-parse", "HEAD").decode().strip()
    task = replace(
        task,
        base_sha=base,
        verify_base_first=True,
        inline_review_threads=2,
        checks=((sys.executable, "-m", "pytest", "-q", "test_number.py"),),
    )

    class RepairGateway(ScriptedGateway):
        def complete(self, **kwargs):
            message = kwargs["messages"][-1]
            if not self.roles:
                assert (
                    "one review concern at a time" in kwargs["messages"][0]["content"]
                )
            if message["role"] == "user" and len(self.roles) == 2:
                assert "test_number.py::test_number" in message["content"]
                assert "Required check 1 exited with status 1" in message["content"]
                assert (
                    "Current draft source near failing pytest frames"
                    in message["content"]
                )
                assert "number.py" in message["content"]
                assert "def current_number():" in message["content"]
                assert "# Candidate source" in message["content"]
                assert "private-" not in message["content"]
                self.code = (
                    "from pathlib import Path; p = Path('number.txt'); "
                    "assert p.read_text() == '5\\n'; p.write_text('4\\n')"
                )
            return super().complete(**kwargs)

    gateway = RepairGateway(
        "from pathlib import Path; Path('number.txt').write_text('5\\n'); "
        "p = Path('number.py'); p.write_text(p.read_text().replace('Baseline', 'Candidate'))"
    )
    verifier = FixtureRunner(source)
    result = Harness2Executor(
        runner, gateway, Limits(model_requests=4), verification_runner=verifier
    ).execute(task, source)

    assert b"+4\n" in result.patch
    assert gateway.roles == ["coding"] * 4
    assert len(verifier.workspaces) == 3
    assert result.refinement.selected_candidate == "repair"
    base, repaired = result.refinement.candidates
    assert base.label == "base"
    assert "test_number.py::test_number" in base.error
    assert "private-" not in base.error
    assert repaired.patch_digest == result.patch_digest


def test_verified_uv_ruff_failure_survives_into_the_repair_turn(fixture):
    from github_harness_agent.executor import AttemptRejected

    task, source, runner = fixture
    launcher = source.parent / "uv"
    launcher.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    launcher.chmod(0o755)
    ruff = source.parent / "ruff"
    ruff.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "if Path('number.txt').read_text() == '5\\n':\n"
        "    print('RUF001 private-message')\n"
        "    print(' --> /tmp/confidential-root/catalog.py:148:22')\n"
        "    print('148 | private-source')\n"
        "    raise SystemExit(1)\n"
    )
    ruff.chmod(0o755)
    task = replace(
        task, checks=((str(launcher), "run", "--no-sync", str(ruff), "check"),)
    )
    draft = "open('number.txt','w').write('5\\n')"
    with pytest.raises(AttemptRejected) as rejected:
        PlainExecutor(runner, ScriptedGateway(draft)).execute(task, source)
    summary = str(rejected.value)
    assert (
        summary
        == "Required check 1 exited with status 1. ruff diagnostics: RUF001 catalog.py:148:22."
    )

    class RepairGateway(ScriptedGateway):
        def complete(self, **kwargs):
            if len(self.roles) == 2:
                assert summary in kwargs["messages"][1]["content"]
                assert "private-" not in kwargs["messages"][1]["content"]
                assert "confidential-root" not in kwargs["messages"][1]["content"]
                self.code = "open('number.txt','w').write('4\\n')"
            return super().complete(**kwargs)

    result = Harness2Executor(runner, RepairGateway(draft)).execute(
        replace(task, verify_base_first=True),
        source,
    )
    assert result.refinement.candidates[0].error == summary
    assert result.refinement.selected_candidate == "repair"


@pytest.mark.parametrize("improves", [False, True])
def test_harness2_review_checkpoint_retains_best_independently_checked_draft(
    fixture, improves
):
    from github_harness_agent.executor import _VerificationRejected

    task, source, runner = fixture
    saved = {}

    def check(expected):
        return (
            sys.executable,
            "-c",
            "from pathlib import Path; "
            f"assert Path('number.txt').read_text().strip() == {expected!r}",
        )

    task = replace(
        task,
        verify_base_first=True,
        checks=(check("5"), check("6")),
        save_checkpoint=lambda draft: saved.update({draft.key: draft}),
    )
    base, repair = ("7", "5") if improves else ("5", "7")

    class RepairGateway(ScriptedGateway):
        def complete(self, **kwargs):
            if len(self.roles) == 2:
                self.code = f"open('number.txt','w').write('{repair}\\n')"
            return super().complete(**kwargs)

    gateway = RepairGateway(f"open('number.txt','w').write('{base}\\n')")
    with pytest.raises(_VerificationRejected):
        Harness2Executor(runner, gateway, Limits(model_requests=4)).execute(
            task, source
        )

    assert b"+5\n" in saved["harness2/base"].patch
    assert not saved["harness2/repair"].patch


def test_harness2_review_retries_missing_regression_coverage_from_safe_draft(fixture):
    task, source, runner = fixture
    task = replace(task, verify_base_first=True, require_new_tests=True)

    class RepairGateway(ScriptedGateway):
        def complete(self, **kwargs):
            message = kwargs["messages"][-1]
            if message["role"] == "user" and len(self.roles) == 2:
                assert "new runnable regression tests" in message["content"]
                self.code = (
                    "from pathlib import Path; "
                    "assert Path('number.txt').read_text() == '4\\n'; "
                    "Path('test_regression.py').write_text("
                    '"from pathlib import Path\\n'
                    "assert Path('number.txt').stat().st_size == 2\\n\")"
                )
            return super().complete(**kwargs)

    gateway = RepairGateway("open('number.txt','w').write('4\\n')")
    result = Harness2Executor(runner, gateway, Limits(model_requests=4)).execute(
        task, source
    )

    assert b"+4\n" in result.patch
    assert b"+++ b/test_regression.py" in result.patch
    assert result.refinement.selected_candidate == "repair"
    assert "new runnable regression tests" in result.refinement.candidates[0].error


@pytest.mark.parametrize(
    "kind", ["legacy_stale", "known_stale", "legacy_extension", "same_seed"]
)
def test_review_repair_checkpoint_keeps_the_current_seed_and_provable_repair_edits(
    fixture, kind
):
    import hashlib

    from github_harness_agent.checkpoint import DraftCheckpoint
    from github_harness_agent.harness2_adapter import BASE_GUIDANCE
    from github_harness_agent.rebase import git

    task, source, runner = fixture
    test_text = (
        "from pathlib import Path\n\n"
        "def test_regression():\n"
        "    assert Path('number.txt').read_text() == '4\\n'\n"
    )

    def patch(files):
        for name, content in files.items():
            (source / name).write_text(content)
        git(source, "add", "-A")
        result = git(
            source, "diff", "--cached", "--binary", "--full-index", task.base_sha
        )
        git(source, "reset", "--hard", task.base_sha)
        return result

    base = patch({"number.txt": "5\n", "test_regression.py": test_text})
    extends = kind in {"legacy_extension", "same_seed"}
    old_files = {
        "number.txt": "5\n" if kind == "legacy_extension" else "4\n",
        "repair_note.txt": "prior work\n",
    }
    if extends:
        old_files["test_regression.py"] = test_text
    saved_repair = DraftCheckpoint(
        "harness2/repair", task.base_sha, patch(old_files), guidance=BASE_GUIDANCE
    )
    if kind in {"known_stale", "same_seed"}:
        saved_repair = replace(
            saved_repair,
            seed_patch_digest=hashlib.sha256(
                base if kind == "same_seed" else b"stale seed"
            ).hexdigest(),
        )
    saved_base = DraftCheckpoint(
        "harness2/base", task.base_sha, base, guidance=BASE_GUIDANCE
    )
    writes = {}
    task = replace(
        task,
        verify_base_first=True,
        require_new_tests=True,
        checkpoints={draft.key: draft for draft in (saved_base, saved_repair)},
        save_checkpoint=lambda draft: writes.update({draft.key: draft}),
    )

    class RepairGateway(ScriptedGateway):
        def complete(self, **kwargs):
            if len(self.roles) == 1:
                assert "Independent validation" in kwargs["messages"][1]["content"]
                self.code = (
                    "from pathlib import Path; "
                    "assert Path('test_regression.py').is_file(); "
                    f"assert Path('repair_note.txt').is_file() == {extends}; "
                    "Path('number.txt').write_text('4\\n'); "
                    "Path('complete.txt').write_text('repair completed\\n')"
                )
            elif len(self.roles) == 2:
                assert kwargs["messages"][-1]["content"].startswith("exit 0")
            return super().complete(**kwargs)

    result = Harness2Executor(runner, RepairGateway(None)).execute(task, source)
    assert b"+++ b/test_regression.py" in result.patch
    assert (b"+++ b/repair_note.txt" in result.patch) == extends
    assert result.refinement.selected_candidate == "repair"
    recorded = writes[saved_repair.key]
    assert recorded.seed_patch_digest == hashlib.sha256(base).hexdigest()
    assert DraftCheckpoint.from_data(recorded.to_data()) == recorded


def test_review_protected_test_rejection_retries_from_last_safe_draft(fixture):
    task, source, runner = fixture
    saved = {}
    task = replace(
        task,
        verify_base_first=True,
        test_update="test_number.py::test_number",
        save_checkpoint=lambda draft: saved.update({draft.key: draft}),
    )

    class RepairGateway(ScriptedGateway):
        def complete(self, **kwargs):
            messages = kwargs["messages"]
            turn = len(self.roles)
            if turn == 1:
                self.code = "open('test_number.py', 'a').write('\\nassert False\\n')"
                kwargs["messages"] = [{"role": "user", "content": "edit"}]
            elif turn == 2:
                assert "restore" in messages[-1]["content"].lower()
                assert "baseline" in messages[-1]["content"]
                self.code = None
            elif turn == 3:
                assert "separate new test module" in messages[-1]["content"]
                self.code = (
                    "from pathlib import Path; assert Path('number.txt').read_text() == '4\\n'; "
                    "assert 'assert False' not in Path('test_number.py').read_text(); "
                    "Path('test_regression.py').write_text('def test_regression():\\n    assert True\\n')"
                )
            return super().complete(**kwargs)

    gateway = RepairGateway("open('number.txt', 'w').write('4\\n')")
    result = Harness2Executor(runner, gateway, Limits(model_requests=5)).execute(
        task, source
    )

    assert gateway.roles == ["coding"] * 5
    assert result.refinement.selected_candidate == "repair"
    assert "baseline" in result.refinement.candidates[0].error
    assert b"test_number.py" not in result.patch
    assert b"test_regression.py" in result.patch
    assert b"test_number.py" not in saved["harness2/base"].patch
