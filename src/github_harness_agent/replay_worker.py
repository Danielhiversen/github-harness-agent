"""A replay arm for the qualification runner. Emit only the verified patch on stdout."""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["plain", "harness2"])
    arm = parser.parse_args().arm
    case = json.load(sys.stdin)
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
    )
    if not snapshot.checks:
        raise ValueError("Required checks are missing")
    result = executor.execute(snapshot, Path(case["checkout"]))
    sys.stdout.buffer.write(result.patch)


if __name__ == "__main__":
    main()
