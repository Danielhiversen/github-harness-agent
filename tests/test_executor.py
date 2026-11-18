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
    with pytest.raises(AttemptRejected, match="Required check failed") as error:
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
