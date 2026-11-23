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
    r"(?:please\s+)?(?:only\s+)?add\s+(?:(?:a|an|one|some)\s+)?(?:new\s+)?"
    r"(?:(?:self[- ]contained|independent|offline|standalone|focused|isolated)[,\s]+){0,4}"
    r"(?:pytest\s+(?:regression\b|tests?\b)|(?:regression\s+)?tests?\b|"
    r"(?:test\s+|regression\s+)?coverage\b)",
    re.IGNORECASE,
)
_NEW_MODULE = r"(?:a\s+)?new\s+(?:(?:pytest\s+)?test\s+)?(?:module|file)"
_TEST_ONLY_MODULE = re.compile(
    rf"(?:(?:a\s+)?test[- ]only\s+(?:change|patch)\s+in\s+{_NEW_MODULE}"
    rf"\s+is\s+(?:sufficient|required)"
    rf"|(?:please\s+)?only\s+add\s+(?:new\s+)?(?:regression\s+)?tests?\s+in\s+{_NEW_MODULE}"
    r"|(?:a\s+)?new\s+(?:pytest\s+)?test\s+(?:module|file)\s+is\s+all\s+that\s+is\s+needed)[.!]?",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\b(?:unless|optional|not necessary)\b|"
    r"\b(?:only\s+if|if\s+(?:needed|possible|appropriate|necessary|feasible|you\s+can))\b",
    re.IGNORECASE,
)
_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)


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


def _review_sentences(body: str):
    body = _HTML_COMMENT.sub(" ", body)
    for line in _review_prose(body):
        line = re.sub(r"(`+).*?\1", " ", line)
        line = re.sub(r"[\"“][^\"”]*[\"”]", " ", line)
        line = re.sub(r"(?<!\w)'[^']*'", " ", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            yield sentence.strip()


_FIXTURE_COVERAGE = re.compile(
    r"^(?:please\s+)?cover\s+.+?\s+with\s+(?:a\s+|an?\s+\w+\s+)?fixtures?\b"
    r"|\b(?:with|add|include|provide|write|use)\s+(?:a|an)\s+"
    r"(?:[A-Za-z-]+\s+){0,3}fixture\s+for\s+\S", re.IGNORECASE,
)
_DECLINED_FIXTURE = re.compile(
    r"\b(?:do not|don't|no need to|avoid|instead of)\s+"
    r"(?:\w+\s+){0,2}(?:add(?:ing)?|includ(?:e|ing)|writ(?:e|ing)|us(?:e|ing)|provid(?:e|ing))\b",
    re.IGNORECASE,
)
_FIXTURE_DIRECTIVE = re.compile(
    r"^(?:please|add|include|provide|write|use|cover|check|verify|exercise|test|ensure|implement|prioriti[sz]e)\b",
    re.IGNORECASE,
)


def fixture_request_sentences(body: str):
    for sentence in _review_sentences(body):
        if (
            _FIXTURE_DIRECTIVE.match(sentence) and _FIXTURE_COVERAGE.search(sentence)
            and not _CONDITIONAL.search(sentence)
            and not _DECLINED_FIXTURE.search(sentence)
        ):
            yield sentence


def requests_new_tests(body: str) -> bool:
    """Recognize an affirmative, direct request in a human review item."""
    return requests_regression_production(body) or any(fixture_request_sentences(body)) or any(
        _ADD_TESTS.match(sentence) and not _CONDITIONAL.search(sentence)
        for sentence in _review_sentences(body)
    )


def requests_test_only_module(body: str) -> bool:
    return requests_new_tests(body) and any(
        _TEST_ONLY_MODULE.fullmatch(sentence) for sentence in _review_sentences(body)
    )


def requests_regression_production(body: str) -> bool:
    return any(
        re.fullmatch(r"(?:please\s+)?update production code and add a new regression test[.!]?", sentence, re.I)
        for sentence in _review_sentences(body)
    )


def authorized_regression_request(feedback: Iterable[Mapping]) -> tuple[str | None, str | None]:
    """Return the pinned revision and the explicitly authorized production path, if any."""
    items = tuple(feedback)
    revisions = set()
    edit_path = None
    directive = "Verify regression against:"
    for item in items:
        if not (
            item.get("key", "").startswith("comment:")
            and item.get("author") and item.get("path") and item.get("thread") and item.get("reply_to")
        ):
            continue
        mixed = requests_regression_production(item["body"])
        test_only = requests_test_only_module(item["body"])
        if not (mixed or test_only):
            continue
        body = _HTML_COMMENT.sub(" ", item["body"])
        for line in _review_prose(body):
            if line.strip().startswith(directive):
                sha = line.strip().removeprefix(directive).strip().lower()
                if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
                    raise ValueError("Verify regression against requires a full commit SHA")
                if mixed:
                    path = item["path"]
                    parts = PurePosixPath(path).parts
                    if (
                        test_only or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,251}\.py", path)
                        or any(part in {"..", ".git", ".github", "tests", "test", "__tests__"} for part in parts)
                        or str(PurePosixPath(path)) != path
                        or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")
                    ):
                        raise ValueError("A mixed regression review must authorize one reviewed Python production file")
                    edit_path = path
                revisions.add(sha)
    if revisions and (len(items) != 1 or len(revisions) != 1):
        raise ValueError("Use one regression verification SHA in a standalone inline review")
    return next(iter(revisions), None), edit_path


def authorized_regression_base(feedback: Iterable[Mapping]) -> str | None:
    return authorized_regression_request(feedback)[0]


def focused_pytest(checks, selectors, report=None, *, verbose=False):
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
        artifacts = (f"--junitxml={report}",) if report is not None else ()
        detail = ("-v",) if verbose else ()
        return (*prefix, "-o", "addopts=", *detail, *artifacts, "--", *selectors)
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


def regression_assertion_report(green: bytes, red: bytes) -> bool:
    """Require an identified passing case to report a call-phase assertion failure."""

    def cases(report):
        result = {}
        for item in ET.fromstring(report).findall(".//testcase"):
            key = item.get("classname"), item.get("name")
            if not all(key) or key in result:
                return {}
            result[key] = item
        return result

    try:
        green_cases = cases(green)
        red_cases = cases(red)
    except ET.ParseError:
        return False
    for name, case in red_cases.items():
        passing = green_cases.get(name)
        if passing is None or any(child.tag in {"failure", "error", "skipped"} for child in passing):
            continue
        if any(child.tag in {"error", "skipped"} for child in case):
            continue
        failure = case.find("failure")
        if failure is not None and re.match(
            r"(?:AssertionError(?::|$)|assert\s)", failure.get("message", "")
        ):
            return True
    return False


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


def has_runnable_test(path: str, source: bytes, *, declarations_only=False) -> bool:
    """Check declarations before running the project's verification commands."""
    if path.endswith("_test.go"):
        return bool(re.search(rb"(?m)^\s*func\s+Test[A-Z0-9_]\w*\s*\(", source))
    if path.endswith((".test.js", ".test.ts", ".spec.js", ".spec.ts")):
        return bool(re.search(rb"(?m)^\s*(?:test|it)(?:\.(?:each|only))?\s*\(", source))
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    if not declarations_only and any(_is_assertion(statement) for statement in tree.body):
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
