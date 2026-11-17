# Isolated repository workers

Build the worker on a disposable Linux execution host with Docker and the
gVisor `runsc` runtime. Install repository-specific tooling and dependencies
in a trusted image derived from the fixture image. Pin the image by digest in
production. Keep registry credentials, provider keys, and GitHub credentials
out of the image and its environment.

```sh
docker build -t gha-fixture-worker:local -f worker/Dockerfile .
docker info --format '{{json .Runtimes}}'
```

Each rollout and the final verification get a separate non-root container.
The executor sends the pinned Git archive over standard input and performs
gateway calls outside the containers. Docker automatically removes each worker
when its task deadline expires, even if the executor exits unexpectedly. The
default Docker network is `none`.
If the deployment needs dependency egress, provision a dedicated Docker network
with a deny-by-default network policy that allows only the approved dependency
sources. The runner expects that policy to be in place when given a network name.

Configure required checks as trusted argument vectors. List any check support
files or directories in the repository's protected path patterns. This covers
test data, wrapper scripts, and tool or CI configuration specific to that
repository. A patch that changes a protected path is rejected before the trusted
checks run. Keep replay acceptance checks outside the rollout containers.

The base image includes Python, Git and tar. The Git snapshot uses
`git archive`, so use repositories without submodule, LFS hydration or
`.gitattributes` export substitutions in the initial pilot. Provision those
inputs explicitly before expanding scope.

## Harness2 integration

The project lockfile pins the upstream Python package. The adapter calls the
upstream sequential core for one refinement step.

```sh
uv sync
uv run pytest -q tests/test_executor.py
```

The fixture uses real Git checkouts and deterministic, in-process gateway
responses. The Harness2 fixture requires that pinned package. To exercise the
Linux runner too, build the image, enable the `runsc` Docker runtime, and run
the same tests with `GHA_DOCKER_TEST_IMAGE=gha-fixture-worker:local`.

The editable component is supplemental coding guidance. The worker policy,
command and patch limits, and trusted checks are enforced by the executor.
Candidate guidance and evidence belong to the task's repository record.
Promotion into shared guidance requires maintainer review.
