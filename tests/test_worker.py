import signal
import subprocess
import sys

import pytest

from github_harness_agent import worker


@pytest.mark.parametrize(
    ("out_bytes", "err_bytes"), [(1025, 0), (800, 225), (800, 224)]
)
def test_completed_process_enforces_combined_output_limit(
    tmp_path,
    monkeypatch,
    out_bytes,
    err_bytes,
):
    release = tmp_path / "release"

    class WritesDuringPoll(subprocess.Popen):
        def poll(self):
            # The child writes and exits after the runner's first chance to
            # sample output sizes, but before poll tells it to stop waiting.
            release.touch()
            return self.wait(timeout=5)

    monkeypatch.setattr(worker.subprocess, "Popen", WritesDuringPoll)
    command = (
        sys.executable,
        "-c",
        "import sys, time; from pathlib import Path\n"
        "while not Path(sys.argv[1]).exists(): time.sleep(0.001)\n"
        f"sys.stdout.buffer.write(b'x' * {out_bytes})\n"
        f"sys.stderr.buffer.write(b'y' * {err_bytes})",
        str(release),
    )
    if out_bytes + err_bytes > 1024:
        with pytest.raises(worker.CommandOutputExceeded):
            worker.run_process(command, 10, lambda: False, cap=1024)
    else:
        result = worker.run_process(command, 10, lambda: False, cap=1024)
        assert result.returncode == 0
        assert result.stdout == b"x" * out_bytes
        assert result.stderr == b"y" * err_bytes


def test_process_group_permission_denial_preserves_limit_error_and_terminates_child(
    monkeypatch,
):
    children = []

    class TrackedProcess(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            children.append(self)

    def group_kill_denied(pid, sig):
        assert pid == children[0].pid
        assert sig == signal.SIGKILL
        raise PermissionError("fixture denies process-group signals")

    monkeypatch.setattr(worker.subprocess, "Popen", TrackedProcess)
    monkeypatch.setattr(worker.os, "killpg", group_kill_denied)
    command = (
        sys.executable,
        "-c",
        "import sys, time; sys.stdout.buffer.write(b'x' * 2048); sys.stdout.flush(); time.sleep(3)",
    )
    with pytest.raises(worker.CommandOutputExceeded):
        worker.run_process(command, 10, lambda: False, cap=1024)
    assert children[0].returncode == -signal.SIGKILL
