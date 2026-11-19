"""Identify tool checks and stat their untracked byproducts in the checkout."""

import inspect
import re
from pathlib import PurePosixPath

CHECK_TOOLS = {
    "jest",
    "mocha",
    "mypy",
    "pytest",
    "ruff",
    "unittest",
    "vitest",
}


def check_label(argv, configured_checks):
    executable = PurePosixPath(argv[0]).name
    if argv in configured_checks:
        return executable

    if executable in {"env", "rtk"}:
        argv = argv[1:]
        if executable == "env":
            while argv and "=" in argv[0] and not argv[0].startswith("-"):
                argv = argv[1:]
        return check_label(argv, ()) if argv else None
    if executable in {"uv", "poetry", "pdm", "pipenv", "hatch"} and argv[1:2] == (
        "run",
    ):
        argv = argv[2:]
        # Skip runner options while looking for the actual check executable.
        while argv and argv[0].startswith("-"):
            argv = (
                argv[2:]
                if argv[0] in {"--directory", "--project", "--python"}
                else argv[1:]
            )
        return check_label(argv, ()) if argv else None
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
        module = argv[2] if len(argv) >= 3 and argv[1] == "-m" else ""
        return module if module in CHECK_TOOLS else None
    if executable in {"cargo", "go", "npm", "pnpm", "yarn"} and argv[1:2] == ("test",):
        return executable
    return executable if executable in CHECK_TOOLS else None


def _stat_files():
    import os
    import sys

    # Paths are checkout-relative bytes from Git, including any whitespace.
    for path in sys.stdin.buffer.read().split(b"\0")[:-1]:
        try:
            info = os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            print()
        else:
            # Ignore atime so reading a generated file keeps its original provenance.
            print(
                f"{info.st_dev} {info.st_ino} {info.st_size} {info.st_mtime_ns} {info.st_ctime_ns}"
            )


STAT_FILES_COMMAND = (
    "python3",
    "-I",
    "-c",
    f"{inspect.getsource(_stat_files)}\n_stat_files()",
)
