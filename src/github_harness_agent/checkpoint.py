"""Bounded draft state that can survive a worker or controller restart."""

import base64
import re
from dataclasses import dataclass


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
