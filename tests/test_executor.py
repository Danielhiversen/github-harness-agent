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

    def complete(self, *, role, messages, tools=(), timeout=None, exclude=()):
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
        def complete(self, *, exclude=(), **kwargs):
            assert exclude == (("other", "different"),)
            return super().complete(**kwargs)

    task, source, runner = fixture
    task = replace(task, excluded_models=(("other", "different"),))
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
    assert result.checks[0].returncode == 0
