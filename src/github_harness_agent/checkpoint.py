"""Bounded draft state that can survive a worker or controller restart."""

import base64
import re
from dataclasses import dataclass

RECOVERY_MODE_THRESHOLD = 2
RECOVERY_GOAL_LIMIT = 24
PRIVATE_PATCH_BYTES = 256 * 1024


@dataclass(frozen=True)
class PrivateRecovery:
    """A single independently checked alternate, outside the primary draft slot."""

    identity: str
    origin: str
    collection: str
    patch: bytes
    preceding_cases: tuple[str, ...]
    cases: tuple[str, ...]
    static: tuple[tuple[str, int], ...] = ()
    dependency_seed: bool = False

    def to_data(self):
        return {
            "identity": self.identity,
            "origin": self.origin,
            "collection": self.collection,
            "patch_b64": base64.b64encode(self.patch).decode("ascii"),
            "preceding_cases": list(self.preceding_cases),
            "cases": list(self.cases),
            "static": [list(check) for check in self.static],
            "dependency_seed": self.dependency_seed,
        }

    @classmethod
    def from_data(cls, data):
        def digest(value):
            return isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{64}", value))

        if not isinstance(data, dict):
            raise ValueError("Invalid private recovery")
        identity, origin, collection = (
            data.get(key) for key in ("identity", "origin", "collection")
        )
        previous, cases = data.get("preceding_cases"), data.get("cases")
        encoded = data.get("patch_b64")
        static = data.get("static")
        dependency_seed = data.get("dependency_seed", False)
        if (
            not all(map(digest, (identity, origin, collection)))
            or not isinstance(encoded, str)
            or len(encoded) > 4 * ((PRIVATE_PATCH_BYTES + 2) // 3)
            or not isinstance(previous, list)
            or not isinstance(cases, list)
            or not 0 < len(cases) < len(previous) <= RECOVERY_GOAL_LIMIT
            or not all(map(digest, (*previous, *cases)))
            or len(set(previous)) != len(previous)
            or len(set(cases)) != len(cases)
            or not set(cases) < set(previous)
            or not isinstance(static, list)
            or len(static) > 2
            or type(dependency_seed) is not bool
        ):
            raise ValueError("Invalid private recovery")
        patch = base64.b64decode(encoded, validate=True)
        if not 0 < len(patch) <= PRIVATE_PATCH_BYTES:
            raise ValueError("Invalid private recovery patch")
        checks = {}
        for check in static:
            if (
                not isinstance(check, list)
                or len(check) != 2
                or not digest(check[0])
                or check[0] in checks
                or type(check[1]) is not int
                or not 1 <= check[1] <= 2048
            ):
                raise ValueError("Invalid private recovery check")
            checks[check[0]] = check[1]
        return cls(
            identity,
            origin,
            collection,
            patch,
            tuple(previous),
            tuple(cases),
            tuple(checks.items()),
            dependency_seed,
        )


@dataclass(frozen=True)
class SolverRecovery:
    identity: str
    goals: tuple[tuple[str, int, int], ...]
    alternate: PrivateRecovery | None = None

    def to_data(self):
        data = {"identity": self.identity, "goals": [list(goal) for goal in self.goals]}
        if self.alternate is not None:
            data["alternate"] = self.alternate.to_data()
        return data

    @classmethod
    def from_data(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Invalid solver recovery state")
        identity, goals = value.get("identity"), value.get("goals")
        try:
            alternate = PrivateRecovery.from_data(value.get("alternate"))
        except ValueError:
            alternate = None
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[a-f0-9]{64}", identity)
            or not isinstance(goals, list)
            or len(goals) > RECOVERY_GOAL_LIMIT
            or not (goals or alternate is not None)
        ):
            raise ValueError("Invalid solver recovery state")
        records = {}
        for goal in goals:
            if not isinstance(goal, list) or len(goal) != 3:
                raise ValueError("Invalid solver recovery goal")
            key, malformed, stalled = goal
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[a-f0-9]{64}", key)
                or key in records
                or any(
                    type(count) is not int or not 0 <= count <= RECOVERY_MODE_THRESHOLD
                    for count in (malformed, stalled)
                )
                or not (malformed or stalled)
            ):
                raise ValueError("Invalid solver recovery goal")
            records[key] = (malformed, stalled)
        return cls(
            identity,
            tuple((key, *counts) for key, counts in records.items()),
            alternate,
        )


@dataclass(frozen=True)
class DraftCheckpoint:
    key: str
    base_sha: str
    patch: bytes
    progress: tuple[str, ...] = ()
    guidance: str | None = None
    last_editor: tuple[str, str] | None = None
    seed_patch_digest: str | None = None

    def to_data(self) -> dict:
        return {
            "key": self.key,
            "base_sha": self.base_sha,
            "patch_b64": base64.b64encode(self.patch).decode("ascii"),
            "progress": list(self.progress),
            "guidance": self.guidance,
            "last_editor": list(self.last_editor) if self.last_editor else None,
            "seed_patch_digest": self.seed_patch_digest,
        }

    @classmethod
    def from_data(cls, data: dict) -> "DraftCheckpoint":
        editor = data.get("last_editor")
        if editor is not None and (
            not isinstance(editor, (list, tuple))
            or len(editor) != 2
            or not all(isinstance(value, str) and value for value in editor)
        ):
            raise ValueError("Invalid saved draft editor")
        seed_digest = data.get("seed_patch_digest")
        if seed_digest is not None and (
            not isinstance(seed_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", seed_digest)
        ):
            raise ValueError("Invalid saved draft seed digest")
        return cls(
            data["key"],
            data["base_sha"],
            base64.b64decode(data["patch_b64"], validate=True),
            tuple(data.get("progress", ())),
            data.get("guidance"),
            (editor[0], editor[1]) if editor is not None else None,
            seed_digest,
        )
