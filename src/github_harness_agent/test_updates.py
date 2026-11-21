"""Parse and verify narrowly scoped amendments to existing Python tests."""

import ast
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath


def failing_pytest_nodes(output: str) -> list[str]:
    """Return bounded test names without parameters, tracebacks or external directories."""
    nodes = []
    for match in re.finditer(
        r"^FAILED (/?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.py)"
        r"((?:::[A-Za-z_][A-Za-z0-9_]*)*::test_[A-Za-z0-9_]+)(?=\[|\s|$)",
        output,
        re.MULTILINE,
    ):
        path = PurePosixPath(match[1])
        prefix = path.name if path.is_absolute() or ".." in path.parts else str(path)
        node = prefix + match[2]
        if len(node) <= 160 and node not in nodes:
            nodes.append(node)
        if len(nodes) == 4:
            break
    return nodes


_DIRECTIVE = re.compile(
    r"Allow test update: ((?:(?:[A-Za-z0-9_][A-Za-z0-9_.-]*)/)*"
    r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.py)"
    r"::(?:([A-Za-z_][A-Za-z0-9_]*)::)?(test_[A-Za-z0-9_]+)"
)
_SKIP_NAMES = {"skip", "skipif", "skipTest", "SkipTest", "xfail", "expectedFailure"}
_ADD_TESTS = re.compile(
    r"(?:please\s+)?add\s+(?:(?:a|an|some)\s+)?(?:new\s+)?"
    r"(?:regression\s+)?(?:tests?\b|(?:test\s+|regression\s+)?coverage\b)",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\b(?:unless|optional|not necessary)\b|"
    r"\b(?:only\s+if|if\s+(?:needed|possible|appropriate|necessary|feasible|you\s+can))\b",
    re.IGNORECASE,
)


def _test_directive(line):
    match = _DIRECTIVE.fullmatch(line)
    if not match:
        return None
    *parents, name = match[1].split("/")
    if (
        name.startswith("test_")
        or name.endswith("_test.py")
        or {"test", "tests"}.intersection(parents)
    ):
        return match[0].removeprefix("Allow test update: ")
    return None


def _review_prose(body):
    fence = ""
    for line in body.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            delimiter = marker[0]
            if not fence:
                fence = delimiter
            elif delimiter.startswith(fence) and not stripped[len(delimiter) :].strip():
                fence = ""
            continue
        if (
            not fence
            and not line.startswith(("    ", "\t"))
            and not stripped.startswith(">")
        ):
            yield line


def authorized_test_update(feedback: Iterable[Mapping]) -> str | None:
    """Read a single unambiguous directive from collected review bodies."""
    directives = {
        selector
        for item in feedback
        if item["key"].startswith("review:") and item["reply_to"] is None
        for line in _review_prose(item["body"])
        if (selector := _test_directive(line))
    }
    return next(iter(directives)) if len(directives) == 1 else None


def requests_new_tests(body: str) -> bool:
    """Recognize an affirmative, direct request in a human review item."""
    body = re.sub(r"<!--.*?(?:-->|\Z)", " ", body, flags=re.DOTALL)
    for line in _review_prose(body):
        line = re.sub(r"(`+).*?\1", " ", line)
        line = re.sub(r"[\"“][^\"”]*[\"”]", " ", line)
        line = re.sub(r"(?<!\w)'[^']*'", " ", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            fixture_coverage = re.match(
                r"(?:please\s+)?cover\s+.+?\s+with\s+(?:a\s+|an?\s+\w+\s+)?fixtures?\b",
                sentence,
                re.IGNORECASE,
            )
            if (
                _ADD_TESTS.match(sentence) or fixture_coverage
            ) and not _CONDITIONAL.search(sentence):
                return True
    return False


def focused_pytest(checks, selectors, report):
    """Reuse the configured pytest launcher with an explicit node selection."""
    for command in checks:
        launcher = 0
        if PurePosixPath(command[0]).name == "uv" and command[1:2] == ("run",):
            launcher = 2
            while launcher < len(command) and command[launcher] in {
                "--no-sync",
                "--offline",
                "--frozen",
                "--locked",
                "--no-project",
                "--",
            }:
                launcher += 1
        if launcher == len(command):
            continue
        executable = PurePosixPath(command[launcher]).name
        if executable == "pytest":
            prefix = command[: launcher + 1]
        elif re.fullmatch(r"python(?:[0-9.]+)?", executable) and command[
            launcher + 1 : launcher + 3
        ] == ("-m", "pytest"):
            prefix = command[: launcher + 3]
        else:
            continue
        return (*prefix, "-o", "addopts=", f"--junitxml={report}", "--", *selectors)
    return None


def passed_pytest_report(report: bytes) -> bool:
    try:
        tests = ET.fromstring(report).findall(".//testcase")
    except ET.ParseError:
        return False
    return bool(tests) and not any(
        result.tag in {"failure", "error", "skipped"}
        for test in tests
        for result in test
    )


def _definition(body, types, name):
    matches = [node for node in body if getattr(node, "name", None) == name]
    if len(matches) != 1 or not isinstance(matches[0], types):
        raise ValueError("Test selector must identify exactly one definition")
    return matches[0]


def _split_test(source: bytes, parts: list[str]):
    tree = ast.parse(source.decode("utf-8"))
    body = tree.body
    if len(parts) == 2:
        body = _definition(body, ast.ClassDef, parts[0]).body
    function = _definition(body, (ast.FunctionDef, ast.AsyncFunctionDef), parts[-1])
    statements = function.body
    first = statements[0].lineno - 1
    lines = source.split(b"\n")
    indentation = lines[first][: statements[0].col_offset]
    if first < function.lineno or not indentation or indentation.strip(b" \t"):
        raise ValueError("Test must have an indented body on its own lines")
    function.body = []
    return ast.dump(tree), lines[:first], lines[function.end_lineno :], statements


def _reference_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _call_name(node):
    return _reference_name(node.func) if isinstance(node, ast.Call) else ""


def _is_assertion(statement):
    if isinstance(statement, ast.Assert):
        return not isinstance(statement.test, ast.Constant)
    if isinstance(statement, ast.Expr) and _call_name(statement.value).startswith(
        "assert"
    ):
        return True
    return isinstance(statement, (ast.With, ast.AsyncWith)) and any(
        _call_name(item.context_expr) in {"raises", "assertRaises", "assertRaisesRegex"}
        for item in statement.items
    )


def has_runnable_test(path: str, source: bytes) -> bool:
    """Check declarations before running the project's verification commands."""
    if path.endswith("_test.go"):
        return bool(re.search(rb"(?m)^\s*func\s+Test[A-Z0-9_]\w*\s*\(", source))
    if path.endswith((".test.js", ".test.ts", ".spec.js", ".spec.ts")):
        return bool(re.search(rb"(?m)^\s*(?:test|it)(?:\.(?:each|only))?\s*\(", source))
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    if any(_is_assertion(statement) for statement in tree.body):
        return True
    bodies = [
        tree.body,
        *(node.body for node in tree.body if isinstance(node, ast.ClassDef)),
    ]
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
        and any(_is_assertion(statement) for statement in ast.walk(node))
        for body in bodies
        for node in body
    )


def valid_test_update(selector: str, baseline: bytes, amended: bytes) -> bool:
    if not _test_directive(f"Allow test update: {selector}"):
        return False
    _, *parts = selector.split("::")
    try:
        old_tree, old_prefix, old_suffix, _ = _split_test(baseline, parts)
        new_tree, new_prefix, new_suffix, statements = _split_test(amended, parts)
    except (SyntaxError, ValueError):
        return False
    if (old_tree, old_prefix, old_suffix) != (new_tree, new_prefix, new_suffix):
        return False
    if not any(_is_assertion(statement) for statement in statements):
        return False
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                return False
            if _reference_name(node) in _SKIP_NAMES:
                return False
    return True
