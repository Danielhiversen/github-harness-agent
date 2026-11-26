"""A replay arm for the qualification runner. Emit only the verified patch on stdout."""

import argparse
import json
import os
import sys
from pathlib import Path

from .executor import (
    DockerRunner,
    Harness2Executor,
    Limits,
    PlainExecutor,
    TaskSnapshot,
)
from .providers import ModelGatewayClient
from .activity import token_counts


def usage_observer(path):
    """Append each call separately so a timed-out replay retains completed work."""
    path.touch()

    def observe(call, phase, role, provider, model, usage=None):
        if phase == "attempt":
            return
        inputs, outputs, complete = token_counts(usage)
        line = json.dumps([call, inputs, outputs, complete]) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)

    return observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["plain", "harness2"])
    arm = parser.parse_args().arm
    case = json.load(sys.stdin)
    meter = os.environ.get("HARNESS_REPLAY_USAGE_FILE")
    backend = Harness2Executor if arm == "harness2" else PlainExecutor
    executor = backend(
        DockerRunner(case["worker_image"]),
        ModelGatewayClient.from_env(),
        Limits(**case.get("limits", {})),
    )
    snapshot = TaskSnapshot(
        case["repository"],
        case["base_sha"],
        case["instructions"],
        tuple(tuple(c) for c in case["checks"]),
        seed=case.get("seed", 0),
        protected_paths=tuple(case.get("protected_paths", ())),
        observe_model=usage_observer(Path(meter)) if meter else None,
    )
    if not snapshot.checks:
        raise ValueError("Required checks are missing")
    result = executor.execute(snapshot, Path(case["checkout"]))
    sys.stdout.buffer.write(result.patch)


if __name__ == "__main__":
    main()
