"""Run repository commands in disposable, network-isolated Linux containers."""

import os
import signal
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from math import ceil


class ExecutionError(RuntimeError):
    pass


class Cancelled(ExecutionError):
    pass


class LimitExceeded(ExecutionError):
    pass


class CommandOutputExceeded(LimitExceeded):
    pass


def run_process(argv, timeout, cancelled, *, cwd=None, data=b"", cap=65536):
    """Bound time and captured output independently of the child process."""
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write(data)
        stdin.seek(0)
        deadline = time.monotonic() + timeout
        with subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        ) as process:
            try:
                while True:
                    if cancelled():
                        raise Cancelled("Task cancelled")
                    if time.monotonic() >= deadline:
                        raise LimitExceeded("Task wall time exceeded")
                    if (
                        os.fstat(stdout.fileno()).st_size
                        + os.fstat(stderr.fileno()).st_size
                        > cap
                    ):
                        raise CommandOutputExceeded(
                            "Command output exceeded the configured size"
                        )
                    if process.poll() is not None:
                        break
                    time.sleep(0.02)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        # Some sandboxes permit signals to our child by PID
                        # while denying process-group signals.
                        process.kill()
        stdout.seek(0)
        stderr.seek(0)
        # A short-lived process can finish writing between the size sample
        # and poll(). Check the combined limit again without reading more than
        # one extra byte or returning truncated output as a success.
        out = stdout.read(cap + 1)
        err = stderr.read(cap + 1 - len(out))
        if len(out) + len(err) > cap:
            raise CommandOutputExceeded("Command output exceeded the configured size")
        return subprocess.CompletedProcess(argv, process.returncode, out, err)


class _DockerWorkspace:
    def __init__(self, name):
        self.name = name

    def command(self, argv, timeout, cancelled, *, data=b"", cap=65536):
        result = run_process(
            ["docker", "exec", "-i", "--user", "65532:65532", "--", self.name, *argv],
            timeout,
            cancelled,
            data=data,
            cap=cap,
        )
        if result.returncode and result.stderr.startswith(
            b"Error response from daemon:"
        ):
            raise ExecutionError("Could not execute a command in the disposable worker")
        return result


class DockerRunner:
    """Run a trusted, locally provisioned image with Git, tar and dependencies.

    An explicit network name assumes deployment-managed egress filtering. The
    default denies egress because gateway calls go through the executor.
    """

    def __init__(self, image: str, network: str = "none", runtime: str = "runsc"):
        if not image or image.startswith("-"):
            raise ValueError("A trusted worker image is required")
        if (
            not network
            or network in {"host", "bridge", "default"}
            or network.startswith("container:")
        ):
            raise ValueError("Use a dedicated filtered network or no network")
        self.image = image
        self.network = network
        self.runtime = runtime

    @contextmanager
    def workspace(self, archive, remaining, cancelled):
        name = f"gha-worker-{uuid.uuid4().hex}"
        # A separate UID protects the container's deadline timer from tool processes.
        lifetime = remaining()
        try:
            try:
                result = run_process(
                    [
                        "docker",
                        "run",
                        "--detach",
                        "--rm",
                        "--init",
                        "--name",
                        name,
                        "--label",
                        "github-harness-agent.worker=true",
                        "--pull",
                        "never",
                        "--runtime",
                        self.runtime,
                        "--network",
                        self.network,
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--read-only",
                        "--log-driver",
                        "none",
                        "--pids-limit",
                        "128",
                        "--memory",
                        "1024m",
                        "--memory-swap",
                        "1024m",
                        "--cpus",
                        "2",
                        "--user",
                        "65533:65533",
                        "--hostname",
                        "worker",
                        "--env",
                        "HOME=/tmp",
                        "--env",
                        "GIT_CONFIG_NOSYSTEM=1",
                        "--env",
                        "GIT_CONFIG_GLOBAL=/dev/null",
                        "--tmpfs",
                        "/work:rw,exec,nosuid,nodev,size=512m,uid=65532,gid=65532,mode=0755",
                        "--tmpfs",
                        "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
                        "--workdir",
                        "/work",
                        "--entrypoint",
                        "/bin/sleep",
                        self.image,
                        str(ceil(lifetime)),
                    ],
                    lifetime,
                    cancelled,
                )
            except FileNotFoundError as exc:
                raise ExecutionError(
                    "Docker CLI is required by the worker runner"
                ) from exc
            if result.returncode:
                raise ExecutionError("Could not start isolated Linux worker")
            workspace = _DockerWorkspace(name)
            result = workspace.command(
                ("tar", "-xf", "-", "--no-same-owner"),
                remaining(),
                cancelled,
                data=archive,
            )
            if result.returncode:
                raise ExecutionError("Could not unpack the pinned repository snapshot")
            yield workspace
        finally:
            try:
                run_process(
                    ["docker", "rm", "--force", "--volumes", name],
                    5,
                    lambda: False,
                )
            except (ExecutionError, OSError):
                pass
