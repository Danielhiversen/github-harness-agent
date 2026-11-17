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
        return subprocess.run(
            argv,
            cwd=self.root,
            input=data,
            capture_output=True,
            timeout=timeout,
            env=dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull),
        )


class ScriptedGateway:
    def __init__(self, code, *, python=sys.executable):
        self.code = code
        self.python = python
        self.roles = []
        self.composition_evidence = {}

    def complete(self, *, role, messages, tools=(), timeout=None):
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


def test_rejects_patch_that_weakens_trusted_check(fixture):
    task, source, runner = fixture
    gateway = ScriptedGateway("open('test_number.py','w').write('pass\\n')")
    with pytest.raises(ExecutionError, match="protected"):
        PlainExecutor(runner, gateway).execute(task, source)


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
