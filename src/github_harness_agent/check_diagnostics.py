"""Summarize recognized check diagnostics without returning raw command output."""

import re
from collections import Counter
from pathlib import PurePosixPath

from .check_artifacts import check_label
from .test_updates import failing_pytest_nodes

_RULE = r"[A-Z]{1,8}[0-9]{2,5}"
_POSITION = r"[1-9][0-9]{0,6}"
_LOCATION = rf"(.{{1,512}}?):({_POSITION}):({_POSITION})"
_RUFF_SHORT = re.compile(rf"^{_LOCATION}: +({_RULE})(?:\s|$)")
_RUFF_HEADER = re.compile(rf"^({_RULE})(?:\s|$)")
_RUFF_ARROW = re.compile(rf"^\s*-->\s+{_LOCATION}\s*$")
_MYPY = re.compile(
    rf"^(.{{1,512}}?):({_POSITION})(?::({_POSITION}))?: (?:error|warning): "
    r".* +\[([a-z][a-z0-9-]{0,47})\]\s*$"
)
_MYPY_LOCATION = re.compile(
    rf"^(.{{1,512}}?):({_POSITION})(?::{_POSITION})?: (?:error|warning): "
)
_PYTEST_LOCATION = re.compile(rf"^(.{{1,512}}?\.py):({_POSITION}):\s")
_SAFE_PATH = re.compile(r"[A-Za-z0-9_. -]+(?:/[A-Za-z0-9_. -]+)*")
_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")


def _format_identifier(code, path, line, column):
    normalized = PurePosixPath(path.replace("\\", "/"))
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or re.match(r"^[A-Za-z]:", path)
    ):
        path = normalized.name
    else:
        path = str(normalized)
    if len(path) <= 120 and normalized.suffix and _SAFE_PATH.fullmatch(path):
        return f"{code} {path}:{line}" + (f":{column}" if column else "")
    return None


def _ruff_matches(output):
    rule = None
    for line in output.splitlines():
        if match := _RUFF_SHORT.match(line):
            path, row, column, code = match.groups()
            yield code, path, row, column
            rule = None
        elif rule and (match := _RUFF_ARROW.match(line)):
            yield rule, *match.groups()
            rule = None
        elif line and not line[0].isspace():
            match = _RUFF_HEADER.match(line)
            rule = match[1] if match else None


def check_details(command, output):
    tool = check_label(tuple(command), ()) if command else None
    if tool not in {"ruff", "mypy", "pytest"}:
        return ""
    if len(output) > 65536:
        output = output[:32768] + "\n" + output[-32768:]
    output = _ANSI_COLOR.sub("", output)
    if tool == "pytest":
        nodes = failing_pytest_nodes(output)
        return (
            "Failing tests: " + ", ".join(nodes) + "."
            if nodes
            else "Check tool: pytest."
        )
    if tool == "mypy":
        matches = (
            _format_identifier(f"[{code}]", path, row, column)
            for line in output.splitlines()
            if (match := _MYPY.match(line))
            for path, row, column, code in (match.groups(),)
        )
    else:
        matches = (_format_identifier(*match) for match in _ruff_matches(output))
    identifiers = []
    for item in matches:
        if item and item not in identifiers:
            identifiers.append(item)
            if len(identifiers) == 4:
                break
    return (
        f"{tool} diagnostics: " + ", ".join(identifiers) + "."
        if identifiers
        else f"Check tool: {tool}."
    )


def check_locations(command, output):
    """Read line locations only from the recognized verifier's diagnostic format."""
    tool = check_label(tuple(command), ()) if command else None
    output = _ANSI_COLOR.sub("", output)
    if tool == "ruff":
        yield from ((path, int(row)) for _, path, row, _ in _ruff_matches(output))
    elif tool in {"mypy", "pytest"}:
        pattern = _MYPY_LOCATION if tool == "mypy" else _PYTEST_LOCATION
        for line in output.splitlines():
            if match := pattern.match(line.lstrip()):
                yield match[1], int(match[2])


def ordinary_static_locations(command, output):
    """Return complete located diagnostics for an ordinary lint or type failure."""
    tool = check_label(command, ())
    output = _ANSI_COLOR.sub("", output)
    if re.search(r"^(?:error:|mypy:|ruff failed\b)", output, re.M):
        return ()
    if tool == "mypy":
        errors = [line for line in output.splitlines() if ": error:" in line]
        matches = [_MYPY.fullmatch(line) for line in errors]
        if not matches or any(match is None for match in matches):
            return ()
        locations = tuple((match[1], int(match[2])) for match in matches)
        summary = re.search(r"^Found (\d+) errors? in \d+ files?\b", output, re.M)
        return locations if summary is None or int(summary[1]) == len(locations) else ()
    if tool == "ruff":
        locations = tuple((path, int(row)) for _, path, row, _ in _ruff_matches(output))
        summary = re.search(r"^Found (\d+) errors?\.", output, re.M)
        if summary is not None and int(summary[1]) == len(locations):
            return locations
    return ()


def ordinary_mypy_errors(command, output):
    """Return normalized locations and exact type messages from a complete mypy report."""
    if check_label(command, ()) != "mypy":
        return ()
    locations = ordinary_static_locations(command, output)
    if not locations:
        return ()
    errors = (line.partition(": error: ")[2] for line in _ANSI_COLOR.sub("", output).splitlines()
              if ": error:" in line)
    return tuple(
        (path.removeprefix("/work/").removeprefix("./"), row, message)
        for (path, row), message in zip(locations, errors, strict=True)
    )


def ordinary_unused_import_failure(command, output):
    """Recognize a complete Ruff report containing only unused imports."""
    if check_label(command, ()) != "ruff" or not ordinary_static_locations(
        command, output
    ):
        return False
    return all(
        code == "F401" for code, *_ in _ruff_matches(_ANSI_COLOR.sub("", output))
    )


def static_error_message(command, output, *, before=""):
    """Return the first new error message, ignoring line shifts and source context."""
    tool = check_label(command, ())

    def messages(text):
        for line in _ANSI_COLOR.sub("", text).splitlines():
            if tool == "mypy" and _MYPY.fullmatch(line) and ": error: " in line:
                yield line.split(": error: ", 1)[1]
            elif tool == "ruff":
                if match := _RUFF_SHORT.match(line):
                    yield line[match.start(4) :].strip()
                elif _RUFF_HEADER.match(line):
                    yield line.strip()

    existing = Counter(messages(before))
    for message in messages(output):
        if not existing[message]:
            return message
        existing[message] -= 1
    return ""


def check_failure_summary(task, checks, *, baseline_failed=False):
    failures = []
    for index, failed in enumerate(checks):
        if not failed.returncode:
            continue
        if failed.command in task.pull_checks:
            label = f"Operator acceptance check {task.pull_checks.index(failed.command) + 1}"
        else:
            label = f"Required check {index % len(task.checks) + 1}"
        summary = f"{label} exited with status {failed.returncode}."
        details = check_details(failed.command, failed.output)
        if details:
            summary += f" {details}"
        failures.append(summary)
    summary = " ".join(failures)
    if baseline_failed:
        summary += " The check also failed on the baseline."
    return summary


def required_review_checks(task):
    """Use the configured check order and append any separate operator checks."""
    return (
        *task.checks,
        *(check for check in task.pull_checks if check not in task.checks),
    )


def check_progress(task, failure):
    """Rank complete trusted checks first, then the presence of required coverage."""
    commands = required_review_checks(task)
    evidence = getattr(failure, "checks", ())
    trusted_checks = evidence[: len(commands)]
    if not commands or tuple(check.command for check in trusted_checks) != commands:
        return None
    if any(not 0 <= check.returncode < 125 for check in evidence):
        return None
    passes = sum(check.returncode == 0 for check in trusted_checks)
    has_coverage = task.require_new_tests and not getattr(
        failure, "missing_regression_tests", False
    )
    return passes, has_coverage
