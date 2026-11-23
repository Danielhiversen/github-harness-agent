"""Incrementally solve Python reviews against operator-supplied pytest checks."""

import ast
import difflib
from pathlib import PurePosixPath
import hashlib
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
import functools
from math import isfinite
from textwrap import dedent

from . import review_recovery
from .check_artifacts import check_label
from .check_diagnostics import (
    check_details,
    check_failure_summary,
    check_locations,
    ordinary_static_locations,
    ordinary_unused_import_failure,
    static_error_message,
)
from .checkpoint import (
    GENERATED_TEST_NAME,
    PRIVATE_PATCH_BYTES,
    RECOVERY_GOAL_LIMIT,
    RECOVERY_MODE_THRESHOLD,
    DraftCheckpoint,
    GeneratedTestSeed,
    PrivateRecovery,
    PrivateCaseTradeoff,
    RedWitness,
    SolverRecovery,
)
from .executor import (
    SYSTEM_PROMPT,
    AttemptRejected,
    CandidateEvidence,
    CheckEvidence,
    RefinementEvidence,
    RouteEvidence,
    _CandidateOutputError,
    _RequestBudgetExceeded,
    _VerificationRejected,
    _combined,
    _protected,
)
from .providers import ProviderExhausted
from .review_composition import (
    compose_line_edits,
    individual_line_edits,
    selective_reverts,
    small_line_edits,
)
from .review_delta_context import interactive_stock_focus
from .review_assertions import assertion_progress, operator_assertions
from .review_delta_context import private_operator_source, reviewed_ancestor_delta
from .review_prefix_replay import observe_prefix_progress
from .review_definitions import (
    CAUSAL_EDIT_GUIDANCE,
    DEFINITION_ENDING,
    DEFINITION_FORMAT,
    NATIVE_DEFINITION_TOOL,
    _symbols,
    apply_definitions,
    apply_native_definition,
    apply_scoped_raw_definition,
    reviewed_nested_helper,
    visible_nested_definitions,
)
from .review_guard_probe import stock_guard_diagnostic
from .review_fixture_audit import audit_fixture_requests, explicit_fixture_requests
from .review_guard_order import swap_traced_filters
from .review_parameter_use import (
    iteration_argument_owner,
    iteration_edit_windows,
    lost_iteration_argument,
)
from .review_round_limit import prepare_round_limit_guard
from .review_semantic_key import guard_traced_counter_once
from .review_guards import guarded_sources
from .review_mock_replay import (
    compact_observations,
    counter_caller_site,
    counter_loop_reuse,
    mock_call_diagnostic,
)
from .review_line_ranges import (
    LINE_RANGE_ENDING,
    NATIVE_LINE_RANGE_TOOL,
    LineRangeSnapshot,
    ScopedRangeRejection,
    full_method_native_tool,
    apply_line_range_edit,
    apply_native_line_range_edit,
    line_range_prompt,
    prepare_line_range_edit,
    apply_offered_line_reply,
    selectable_line_tool,
    prepare_loop_line_range_edit,
    prepare_window_line_range_edit,
)
from .review_planner import (
    apply_coding_advice,
    localized_retry_symbol,
    pinned_counter_location,
    request_coding_advice,
    private_method_planning_assist,
    pinned_stock_restore_seed,
)
from .review_recovery import operator_acceptance_context
from .review_solver_context import (
    OperatorTestAssembly,
    _pytest_fixtures,
    _rewritten_assertion,
    extended_pytest_context,
    focused_check_excerpt,
    focused_source_excerpt,
    imported_model_declarations,
    operator_api_contract,
    operator_test_assembly,
    production_protocol_example,
    direct_caller_names,
    prior_generated_fixture_context,
    prior_generated_test_seed,
    repository_pytest_context,
    stock_assertion_progress,
    stock_assertion_failure_line,
    stock_failure_site,
    stock_integer_assertion_failure_line,
    stock_missing_count_after_reorder,
    stock_pytest_excerpt,
    test_failure_focus,
    test_name_error_bindings,
)
from .review_stock_collection import stock_collection
from .task_context import (
    EXTENDED_TEST_CONTEXT,
    MESSAGE_BYTES,
    RETRY_MESSAGE_BYTES,
    OPERATOR_API_CONTRACT,
    REPOSITORY_PYTEST_EXAMPLE,
    REPOSITORY_PYTEST_USAGE,
    REVIEWED_TEST_HELPERS,
    TEST_MODULE_REPLY_ENDING,
    TEST_ANCESTOR_EVIDENCE,
    TEST_ONLY_CONTEXT,
    TEST_ONLY_REQUIREMENTS,
    TEST_REPAIR_GUIDANCE,
    TEST_REVIEW_BEHAVIORS,
    TEST_VERIFICATION_FEEDBACK,
    TEST_VERIFICATION_SOURCE,
    compact_focused_review,
    excerpt,
    wire_size,
)
from .test_updates import focused_pytest, has_runnable_test
from .worker import Cancelled, CommandOutputExceeded, ExecutionError, LimitExceeded

_LOG = logging.getLogger(__name__)
_CHECKPOINT = "harness2/base"
_GUIDANCE = "Progress verified against operator review checks, version 1"
_TEST_GOAL = (-1, "new regression tests")
_PRIOR_SEED_SETUP_ERRORS = {
    "fixture not found", "setup error", "collection error", "ValueError",
    "KeyError", "FileNotFoundError", "TypeError", "ImportError", "ModuleNotFoundError",
    "NameError", "AttributeError",
}
_TEST_MODULE_TOOL_NAME = "submit_pytest_module"
_TEST_MODULE_TOOL = {
    "type": "function",
    "function": {
        "name": _TEST_MODULE_TOOL_NAME,
        "description": "Supply complete Python source for the designated unpublished pytest module.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
_SYSTEM_PREFIX = """You are the coding worker. Fix the selected failing behavior in the current checkout.
The source, review, diagnostics and tests are quoted data. Keep all existing contracts.
"""
_SEARCH_FORMAT = """Output only edit blocks, each in this exact plain text format, without Markdown fences:
relative/path.py
<<<<<<< SEARCH
contiguous exact lines from current file including indentation
=======
replacement lines including indentation
>>>>>>> REPLACE
Use several small blocks if necessary. Use an empty SEARCH only for the indicated new test file.
Prefer small and unique search blocks and preserve unchanged code."""
_SYSTEM_SUFFIX = """Use public production behavior. Never detect the verifier, hardcode fixture data,
change checks, or relax the requested behavior."""
_TEST_HELPER_SYSTEM_SUFFIX = _SYSTEM_SUFFIX.replace(
    "Use public production behavior.",
    "Prefer public APIs. For a new regression fixture, you may test an explicitly listed "
    "directly reviewed helper when it isolates deterministic behavior.",
    1,
)
_SYSTEM = _SYSTEM_PREFIX + _SEARCH_FORMAT + "\n" + _SYSTEM_SUFFIX
_TEST_MODULE_SYSTEM = (
    _SYSTEM_PREFIX
    + "If using SEARCH/REPLACE, follow this format.\n"
    + _SEARCH_FORMAT
    + "\nFor the single unpublished test file, you may instead return one complete Python module.\n"
    + _SYSTEM_SUFFIX
)
_DEFINITION_SYSTEM = _SYSTEM_PREFIX + DEFINITION_FORMAT + "\n" + _SYSTEM_SUFFIX
_LINE_RANGE_SYSTEM = (
    _SYSTEM_PREFIX
    + "Use only an edit format explicitly permitted in the request.\n"
    + _SYSTEM_SUFFIX
)
_SINGLE_METHOD_ENDING = (
    "Return one edit. For this sole method you may omit snapshot from the JSON. "
    "Alternatively return up to six nonoverlapping exact SEARCH/REPLACE blocks for this method, or its complete Python definition with its "
    "exact header and a small body change. The same scope and changed-row limits apply."
)
_BLOCK = re.compile(
    r"^([^\n`]+\.py)\n<<<<<<< SEARCH\n(.*?)^=======\n(.*?)^>>>>>>> REPLACE[ \t]*(?:\n|$)",
    re.M | re.S,
)
_MARKER = re.compile(r"^(?:<{5,}|={5,}|>{5,})(?:[ \t]|$)", re.M)
_PYTHON_FENCE = re.compile(
    r"(?P<fence>`{3,}|~{3,})(?:python|py)?[ \t]*\n(?P<code>.*?)\n(?P=fence)",
    re.S | re.I,
)
_FENCE_LINE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.M)
_FILE_HEADER = re.compile(
    r"^[ \t]*#?[ \t]*(?:file(?:name)?|path):[ \t]*[`'\"]?\S+\.py[`'\"]?[ \t]*$",
    re.M | re.I,
)
_TEST_RESULT = re.compile(r"^(FAILED|ERROR) ([^\s]+::[^\s]+)", re.M)
_NAMED_PYTEST_PASS = re.compile(
    r"^(\S+\.py::\S+) PASSED(?:[ \t]+\[[ 0-9]+%\])?[ \t]*$", re.M
)
_NAMED_PYTEST_FAIL = re.compile(
    r"^FAILED (\S+\.py::\S+)(?:[ \t]+- [^\n]*)?[ \t]*$", re.M
)
_TEST_SECTION = re.compile(r"^_{5,} (.+?) _{5,}$", re.M)
_PYTEST_SECTION = re.compile(r"^={5,} ([^\n]+?) ={5,}$", re.M)
_PYTEST_CRASH_LINE = re.compile(r"^\S[^\n]*?\.py:[1-9][0-9]*: .+$")
_TEST_ERROR_KINDS = re.compile(
    r"\b(?:AssertionError|AttributeError|FileNotFoundError|ImportError|IndentationError|"
    r"IndexError|KeyError|ModuleNotFoundError|NameError|RuntimeError|SyntaxError|"
    r"TypeError|ValueError)\b"
)
_PYTEST_GENERIC_HEADING = re.compile(r"ERROR (collecting \S+|at setup of .+)")
_PYTEST_REPORT_PATH = r"[/\w.-]{1,480}\.py"
_PYTEST_GENERIC_FRAME = re.compile(rf"^({_PYTEST_REPORT_PATH}):([1-9][0-9]{{0,6}})(?::(?:\s+(.+))?)?$")
_PYTEST_FILE_FRAME = re.compile(
    rf'^(?:File "({_PYTEST_REPORT_PATH})"|file ({_PYTEST_REPORT_PATH})), '
    r"line ([1-9][0-9]{0,6})(?:, in .*)?$"
)
_PYTEST_GENERIC_SUMMARY = re.compile(rf"^ERROR ({_PYTEST_REPORT_PATH})(?: - (.+))?$")
_PYTEST_CAPTURED = re.compile(r"^-{5,} Captured (?:stdout|stderr|log)(?: .+)? -{5,}$")
_PYTEST_EXCEPTION = re.compile(r"^(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*(?:Error|Exception)|Failed)(?::|$)")
_PYTEST_FIXTURE_MISSING = re.compile(r"""^fixture ['"][A-Za-z_]\w{0,120}['"] not found\b""")
_PYTEST_CLI_ERROR = re.compile(r"^ERROR: (?:file or directory not found|not found|usage):?\s+.+$")
_DRAFT_EXCEPTION = re.compile(
    r"^(?:E[ \t]+|(?:FAILED|ERROR) \S+ - |\S+\.py:[1-9][0-9]*: )"
    r"([\w.]*(?:Error|Exception):[ \t]*\S[^\n]*)$",
    re.M,
)
_PYTEST_TOTALS = re.compile(r"(?:\d+ [a-z]+, )*\d+ [a-z]+ in \d+(?:\.\d+)?s(?: \([\d:]+\))?")
_STOCK_REPAIR_CASE_LIMIT = 4
_DEPENDENCY_SEED_CASE_LIMIT = 6
_DEPENDENCY_SEED_PROBE_SECONDS = 120
_STOCK_REPAIR_STATIC_LIMIT = 2
_STOCK_REPAIR_RETRIES = 2
_STOCK_HEDGE_DELAY = 60
_INTERACTIVE_MAX_TURNS = 12
_INTERACTIVE_MAX_TOOLS = 24
_INTERACTIVE_RESERVE_SECONDS = 60
_INTERACTIVE_CAPTURE_RESERVE_SECONDS = 5
_INTERACTIVE_INSTRUCTIONS_BYTES = 4100
_GUARD_SECONDS = 90
_GUARD_MIN_REMAINING = 45
_COMPOSITION_MIN_REMAINING = 60
_STOCK_HUNK_PROBES = 3
_STOCK_HUNK_SECONDS = 12
_MODEL_FAILURE_THRESHOLD = 2
_LOOP_PRECISION_MODEL_SECONDS = 35
_LOOP_PRECISION_RESERVE_SECONDS = 65
_LOOP_PRECISION_MIN_REMAINING = (
    _LOOP_PRECISION_MODEL_SECONDS + _GUARD_SECONDS + _LOOP_PRECISION_RESERVE_SECONDS
)
_LONG_LOCALIZED_DEFINITION_BYTES = 6 * 1024
_WORD = re.compile(r"[a-z]{3,}")
_STOP = set(
    "self test check review request fixture calls builder return catalog assert tests with this from that they when have with which should".split()
)


def words(text):
    return set(_WORD.findall(text.casefold())) - _STOP


def apply_blocks(response, sources, new_path):
    response = (response or "").strip().replace("\r\n", "\n")
    # Common Markdown fencing is harmless outside an edit.
    response = re.sub(r"(?m)^```[\w.-]*[ \t]*(?:\n|$)", "", response)
    # Normalize a filename that a provider styled as Markdown. Only a header
    # immediately preceding a SEARCH marker is considered.
    response = re.sub(
        r"(?mi)^(?:file:[ \t]+)?[`*]*([^\s`*]+\.py)[`*]*:?[ \t]*(?=\n<<<<<<< SEARCH$)",
        r"\1",
        response,
    )
    if len(response) > 80000:
        raise ValueError("Response exceeded the edit limit")
    result = sources.copy()
    end = 0
    for match in _BLOCK.finditer(response + "\n"):
        if _MARKER.search(response[end : match.start()]):
            raise ValueError(
                "Return edit blocks only, with the exact relative filename"
            )
        path, old, new = match.groups()
        if _MARKER.search(old) or _MARKER.search(new):
            raise ValueError("Incomplete or nested SEARCH/REPLACE block")
        path = path.strip()
        if path not in sources:
            raise ValueError(
                "File is outside the editable source list. Use the exact listed checkout path."
            )
        source = result[path]
        if not old or (path == new_path and not source and not old.strip()):
            if path != new_path or source:
                raise ValueError("Empty SEARCH is only permitted for the new test file")
            result[path] = new
        else:
            if source.count(old) != 1:
                raise ValueError(
                    f"SEARCH for {path} must match exactly once: {old[:180]}"
                )
            result[path] = source.replace(old, new, 1)
        end = match.end()
    if not end or _MARKER.search(response[end:]) or sources == result:
        raise ValueError("No complete edit blocks changed the checkout")
    for path, source in result.items():
        if source != sources[path]:
            if len(source.encode()) > 256 * 1024:
                raise ValueError("Edited file exceeds the source limit")
            try:
                ast.parse(source, filename=path)
            except (SyntaxError, ValueError, RecursionError) as error:
                raise ValueError(f"{path}: invalid Python: {error}") from None
    return result


def _complete_test_module(response, path, limit):
    if not isinstance(response, str):
        return None
    try:
        maximum = min(limit, 80000)
        if len(response.encode()) > maximum:
            return None
        response = response.replace("\r\n", "\n")
        stripped = response.strip()
        fenced = _PYTHON_FENCE.fullmatch(stripped)
        if not fenced and len(markers := list(_FENCE_LINE.finditer(stripped))) == 2:
            start, close = (marker.start() for marker in markers)
            end = stripped.find("\n", close)
            end = len(stripped) if end < 0 else end
            outside = stripped[:start] + stripped[end:]
            if (
                len(outside) <= 600 and len(outside.splitlines()) <= 6
                and not _MARKER.search(outside) and ".py" not in outside
                and not re.search(r"(?m)^\s*(?:diff |@@|REPLACE |--- |\+\+\+ )", outside)
            ):
                fenced = _PYTHON_FENCE.fullmatch(stripped[start:end].strip())
        source = (fenced["code"] if fenced else response).rstrip("\n") + "\n"
        data = source.encode()
        if (
            len(data) > maximum or _MARKER.search(source)
            or _FENCE_LINE.search(source) or _FILE_HEADER.search(source)
        ):
            return None
        tree = ast.parse(source, filename=path)
        for index, statement in enumerate(tree.body):
            if isinstance(statement, ast.Expr) and not (
                isinstance(statement.value, ast.Call)
                or (
                    index == 0
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ):
                return None
        compile(tree, path, "exec", dont_inherit=True)
        return source if has_runnable_test(path, data, declarations_only=True) else None
    except (SyntaxError, ValueError, RecursionError):
        return None


def _retry_test_definition(response, draft, path, limit):
    """Apply an unlabelled reply only when it identifies a single draft function."""
    if not isinstance(response, str) or not draft:
        return None
    try:
        if len(response.encode()) > min(limit, 80_000):
            return None
        code = response.replace("\r\n", "\n").strip()
        if fence := _PYTHON_FENCE.fullmatch(code):
            code = fence["code"]
        if _MARKER.search(code) or _FENCE_LINE.search(code) or _FILE_HEADER.search(code):
            return None
        body = ast.parse(code, filename=path).body
        if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        replacement = body[0]
        current = ast.parse(draft, filename=path).body
    except (SyntaxError, ValueError, RecursionError):
        return None

    matches = [
        node for node in current
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == replacement.name
    ]
    if len(matches) != 1 or type(matches[0]) is not type(replacement):
        return None

    symbol = (path, replacement.name)
    result = apply_definitions(
        f"REPLACE {path}::{replacement.name}\n{code}",
        {path: draft}, only=symbol, single_function=True,
    )
    try:
        compile(result[path], path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError(f"{path}: invalid Python: {error}") from None
    return result


def _wrapped_test_search(code, draft, path, labeled, limit):
    if labeled and code.startswith("<<<<<<< SEARCH\n"):
        code = f"{path}\n{code}"
    block = _BLOCK.fullmatch(code)
    if not block or block[1] != path or not block[2].strip():
        raise ValueError
    try:
        result = apply_blocks(code, {path: draft}, path)
    except ValueError:
        if replacement := _replace_stale_test_module(code, draft, path, limit):
            return replacement
        raise
    original = ast.parse(draft, filename=path).body
    candidate = ast.parse(result[path], filename=path).body
    # The exact search must lie inside one top-level function body. Comparing its
    # siblings also detects edits that escape the body through a dedent or string.
    offset = draft.index(block[2])
    first_line = draft.count("\n", 0, offset) + 1
    last_line = first_line + block[2].rstrip("\n").count("\n")
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    if len(original) != len(candidate):
        raise ValueError
    inside_function = False
    for old, new in zip(original, candidate, strict=True):
        if isinstance(old, functions) and old.lineno < first_line <= last_line <= old.end_lineno:
            inside_function = True
            if type(new) is not type(old) or new.name != old.name:
                raise ValueError
        elif ast.dump(old) != ast.dump(new):
            raise ValueError
    if not inside_function:
        raise ValueError
    return result


def _test_module_declarations(body, *, include_helpers=False):
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    owners = [
        ("", body),
        *((node.name, node.body) for node in body if isinstance(node, ast.ClassDef)),
    ]
    declarations = Counter(
        (owner, node.name)
        for owner, statements in owners
        for node in statements
        if isinstance(node, functions) and (include_helpers or node.name.startswith("test"))
    )
    declarations.update(
        ast.dump(node) for node in body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    if include_helpers:
        declarations.update(("class", node.name) for node in body if isinstance(node, ast.ClassDef))
        declarations.update(
            ("fixture", owner, name, automatic)
            for owner, statements in owners
            for name, _, automatic in _pytest_fixtures(statements)
        )
    return declarations


def _test_module_shape(tree):
    """Keep declarations and module-level statements while ignoring function bodies."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = []
    return ast.dump(tree)


def _replace_stale_test_module(response, draft, path, limit):
    """Recover a failed exact search only when both sides are whole test modules."""
    if not isinstance(response, str) or not draft:
        return None
    try:
        if len(response.encode()) > min(limit, 80_000):
            return None
        code = response.replace("\r\n", "\n").strip()
        if fence := _PYTHON_FENCE.fullmatch(code):
            code = fence["code"]
        block = _BLOCK.fullmatch(code)
        if not block or block[1] != path or not block[2].strip() or block[2] in draft:
            return None
        sources = [_complete_test_module(source, path, limit) for source in (draft, block[2], block[3])]
        if not all(sources):
            return None
        draft_module, search_module, new_module = sources
        current, searched, candidate = (
            ast.parse(source, filename=path) for source in sources
        )
        if (
            min(len(tree.body) for tree in (current, searched, candidate)) < 2
            or new_module in (draft_module, search_module)
            or not _test_module_declarations(current.body, include_helpers=True)
            <= _test_module_declarations(candidate.body, include_helpers=True)
            or _test_module_shape(current) != _test_module_shape(searched)
        ):
            return None
        return {path: new_module}
    except (ValueError, RecursionError):
        return None


def _wrapped_test_repair(response, draft, path, limit):
    """Recover a single edit from a labeled or prose-wrapped draft repair."""
    if not isinstance(response, str):
        return None
    text = response.replace("\r\n", "\n").strip()
    fences = list(_FENCE_LINE.finditer(text))
    if not fences:
        return None
    if len(fences) != 2:
        raise ValueError("Pytest repair declined (ambiguous-fences)")
    start, close = (fence.start() for fence in fences)
    end = text.find("\n", close)
    end = len(text) if end < 0 else end
    before, after = text[:start].strip(), text[end:].strip()
    if not before and not after:
        return None
    if not after and re.fullmatch(r"REPLACE \S+\.py::[\w.]+", before):
        return None  # The canonical definition header may precede its fenced body.

    if len(response.encode()) > min(limit, 80_000):
        raise ValueError("Pytest repair declined (response-size)")
    outside = "\n".join(part for part in (before, after) if part)
    if len(outside) > 600 or len(outside.splitlines()) > 6:
        raise ValueError("Pytest repair declined (wrapper-size)")
    target = re.escape(path)
    labeled = False
    for line in outside.splitlines():
        if re.fullmatch(
            rf"(?:(?i:file|filename|path):[ \t]*)?(?:{target}|`{target}`|\*\*{target}\*\*):?",
            line.strip(),
        ):
            if labeled:
                raise ValueError("Pytest repair declined (duplicate-wrapper-label)")
            labeled = True
        elif re.search(r"(?i)\.py|(?:[\w.-]+/)+[\w.-]+", line):
            raise ValueError("Pytest repair declined (wrapper-path)")
        elif _MARKER.search(line.strip()) or re.match(
            r"\s*(?:diff |@@|REPLACE\b|IMPORTS\b|--- |\+\+\+ |(?:file|filename|path):)",
            line, re.I,
        ):
            raise ValueError("Pytest repair declined (wrapper-operation)")

    fence = _PYTHON_FENCE.fullmatch(text[start:end].strip())
    if not fence:
        raise ValueError("Pytest repair declined (fence-format)")
    code = fence["code"]
    try:
        if code.startswith("REPLACE "):
            result = apply_definitions(code, {path: draft}, single_function=True)
        elif _MARKER.search(code):
            result = _wrapped_test_search(code, draft, path, labeled, limit)
        else:
            result = _retry_test_definition(code, draft, path, limit)
            if result is None and (module := _complete_test_module(code, path, limit)):
                proposed = ast.parse(module, filename=path).body
                original = ast.parse(draft, filename=path).body
                if (
                    len(proposed) > 1
                    and _test_module_declarations(original) <= _test_module_declarations(proposed)
                ):
                    result = {path: module}
        if result is not None:
            compile(result[path], path, "exec", dont_inherit=True)
            return result
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("Pytest repair declined (invalid-wrapped-edit)") from None
    raise ValueError("Pytest repair declined (unrecognized-wrapped-edit)")


def _apply_test_reply(response, sources, path, *, allow_module, failed_draft, limit):
    if allow_module and failed_draft and (draft := sources.get(path)):
        if wrapped := _wrapped_test_repair(response, draft, path, limit):
            return wrapped
    try:
        return apply_blocks(response, sources, path)
    except ValueError:
        if allow_module and failed_draft:
            if replacement := _replace_stale_test_module(response, sources.get(path), path, limit):
                return replacement
            retry = _retry_test_definition(response, sources.get(path), path, limit)
            if retry is not None:
                return retry
        if allow_module and (module := _complete_test_module(response, path, limit)):
            return {path: module}
        if allow_module and failed_draft and re.search(r"(?m)^REPLACE ", response or ""):
            return apply_definitions(
                response, {path: sources[path]}, single_function=True
            )
        raise


def _apply_native_test_module(calls, path, limit):
    try:
        if len(calls) != 1 or calls[0]["type"] != "function":
            raise ValueError
        function = calls[0]["function"]
        arguments = function["arguments"]
        if (
            function["name"] != _TEST_MODULE_TOOL_NAME
            or not isinstance(arguments, str)
            or len(arguments.encode()) > 16_384
        ):
            raise ValueError
        payload = json.loads(arguments, object_pairs_hook=review_recovery._unique_object)
        if type(payload) is not dict or payload.keys() != {"code"}:
            raise ValueError
        if module := _complete_test_module(payload["code"], path, limit):
            return {path: module}
    except (IndexError, KeyError, TypeError, ValueError, RecursionError):
        pass
    raise ValueError("Expected one complete pytest module in the code argument")


def patch_for(original, sources):
    diff = []
    for path, source in sources.items():
        before = original[path]
        if before != source:
            diff.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    source.splitlines(True),
                    fromfile="a/" + path if before else "/dev/null",
                    tofile="b/" + path,
                )
            )
    return "".join(diff).encode()


def _test_detail(output, node):
    name = node.split("::", 1)[1].replace("::", ".")
    heading = re.compile(rf"(?:ERROR at \w+ of )?{re.escape(name)}")
    sections = _TEST_SECTION.split(output)
    for title, body in zip(sections[1::2], sections[2::2], strict=True):
        if heading.fullmatch(title):
            frame_lines = []
            lines = []
            for line in body.splitlines():
                if line.startswith("-----"):
                    break
                if _PYTEST_CRASH_LINE.fullmatch(line.lstrip()):
                    frame_lines.append(line)
                elif (
                    title.startswith("ERROR at setup of ")
                    and _PYTEST_FILE_FRAME.fullmatch(line.strip())
                ):
                    frame_lines.append(line.strip())
                elif line.startswith(("E ", "> ")):
                    lines.append(line)
            lines = [*frame_lines[:4], *lines]
            if lines:
                return excerpt("\n".join(lines), 1600)
    return next((line for line in output.splitlines() if node in line), node)


def _unpublished_case_detail(output, node, detailed):
    """Use terminal assertion messages and literal operands, without quoting verifier source."""
    sections = _PYTEST_SECTION.split(output)
    footer = dict(zip(sections[1::2], sections[2::2], strict=True)).get("short test summary info", "")
    summary = next((
        line.partition(" - ")[2] for line in footer.splitlines()
        if (match := _TEST_RESULT.match(line)) and match[2] == node and " - " in line
    ), "")
    messages = []
    for line in (*detailed.splitlines(), summary):
        # Only pytest's reported exception, crash line, or rewritten expression.
        # The source line ("> assert ...") and captured test output are excluded.
        reason = re.sub(r"^(?:E[ \t]+|\S[^\n]*?\.py:[1-9][0-9]*: )", "", line).strip()
        if reason.startswith("AssertionError: assert "):
            reason = reason.removeprefix("AssertionError: ")
        if reason.startswith("AssertionError: ") and not reason.endswith(("...", "…")):
            messages.append(excerpt(reason, 350))
        elif reason.startswith("assert ") and len(reason) <= 160 and _scalar_assertion(reason[7:]):
            messages.append(reason)
    return node + ("\nObserved pytest failure: " + "; ".join(dict.fromkeys(messages))[:500] if messages else "")


def _operator_diagnostic_replay(command, node):
    selection = node.partition("::")[2]
    if not selection or len(selection) > 300:
        return None
    allowed = []
    for arg in command:
        path = arg.partition("::")[0]
        if not path.startswith(review_recovery.OPERATOR_CHECK_ROOT + "/"):
            continue
        if _trusted_operator_node(node, (arg,), {path: {selection.partition("[")[0]}}):
            allowed.append(path + "::" + selection)
    if len(allowed) != 1:
        return None
    return focused_pytest((command,), allowed, verbose=True)


def _pytest_reason_kind(reason):
    if not reason:
        return ""
    if re.match(r"^(?:Failed:\s+)?DID NOT RAISE(?:\s|$)", reason):
        return "DID NOT RAISE"
    if _PYTEST_FIXTURE_MISSING.match(reason):
        return "fixture not found"
    match = _PYTEST_EXCEPTION.match(reason)
    if match:
        kind = match[1]
        return kind if _TEST_ERROR_KINDS.fullmatch(kind) else "pytest error"
    return ""


def _pytest_report_diagnostic(title, body):
    heading = title if _PYTEST_GENERIC_HEADING.fullmatch(title) else ""
    location, reason, kind = "", "", ""
    for line in body.splitlines():
        line = line.strip()
        if _PYTEST_CAPTURED.fullmatch(line):
            break
        if frame := _PYTEST_GENERIC_FRAME.fullmatch(line):
            location = f"{frame[1]}:{frame[2]}"
            message = frame[3] or ""
        elif native := _PYTEST_FILE_FRAME.fullmatch(line):
            location = f"{native[1] or native[2]}:{native[3]}"
            continue
        elif line.startswith("E "):
            message = line[1:].lstrip()
        else:
            continue
        if current := _pytest_reason_kind(message):
            # Preserve the more specific pytest spelling if the crash row
            # repeats only the exception class.
            if not reason or ":" in message:
                kind, reason = current, message
    if not kind and heading:
        kind = "collection error" if heading.startswith("ERROR collecting ") else "setup error"
    detail = "\n".join(part for part in (excerpt(heading, 150), location, excerpt(reason, 420)) if part)
    return detail, kind


def _generic_verifier_failure(command, returncode, output):
    """Select report metadata before bounding a non-named verifier failure."""
    if check_label(command, ()) != "pytest":
        return excerpt(output, 3000), ""
    output_text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    sections = _PYTEST_SECTION.split(output_text)
    fallback = ("", "")
    for title, content in zip(sections[1::2], sections[2::2], strict=True):
        if title in {"ERRORS", "FAILURES"}:
            reports = _TEST_SECTION.split(content)
            for heading, body in zip(("", *reports[1::2]), reports[::2], strict=True):
                detail, kind = _pytest_report_diagnostic(heading, body)
                if kind and kind not in {"collection error", "setup error"}:
                    return detail, kind
                if kind and not fallback[1]:
                    fallback = detail, kind
        elif title == "short test summary info":
            for line in content.splitlines():
                if summary := _PYTEST_GENERIC_SUMMARY.fullmatch(line):
                    kind = _pytest_reason_kind(summary[2])
                    if kind:
                        return excerpt(line, 700), kind
                    if not fallback[1]:
                        fallback = excerpt(line, 700), "collection error"
    if fallback[1]:
        return fallback
    if returncode == 4 and len(sections) == 1:
        for line in output_text.splitlines():
            if _PYTEST_CLI_ERROR.fullmatch(line):
                return excerpt(line, 700), "usage error"
    return excerpt(output, 3000), ""


def _test_diagnostic_label(command, returncode, categories):
    label = check_label(command, ())
    if label == "pytest":
        exit_kind = {
            1: "failed tests", 2: "interrupted", 3: "internal error",
            4: "usage error", 5: "no tests collected",
        }.get(returncode, "execution error")
    else:
        label = label if label in {"ruff", "mypy"} else "configured"
        exit_kind = "execution error" if returncode < 0 or returncode >= 125 else "nonzero exit"
    return f"{label}/{exit_kind}: {', '.join(sorted(categories)) or 'check failure'}"


def _named_test_diagnostic_kind(detail):
    if _TEST_RESULT.match(detail):
        first = detail.partition("\n")[0]
        return _pytest_reason_kind(first.partition(" - ")[2])
    return _pytest_report_diagnostic("", detail)[1]


def _failed_test_feedback(failures, path):
    cases = {}
    for (_, node), detail in failures.items():
        if node.removeprefix("./").startswith(path + "::"):
            cases.setdefault(node, detail)
    if not cases:
        return "\n".join(
            f"{name}: {excerpt(detail, 1000)}"
            for (_, name), detail in list(failures.items())[:4]
        )

    signatures = {
        node: match[1].strip()
        for node, detail in cases.items()
        if (match := _DRAFT_EXCEPTION.search(detail)) and not match[1].rstrip().endswith("...")
    }
    counts = Counter(signatures.values())
    selected = max(cases, key=lambda node: counts[signatures.get(node)])
    lines = [f"Selected failing case: {selected}", excerpt(cases[selected], 800)]
    if (count := counts[signatures.get(selected)]) > 1:
        lines.append(
            f"{count} cases report the same exception and message. "
            "They may share a fixture or setup error."
        )
    remaining = [node for (_, node) in failures if node != selected]
    if remaining:
        lines.append("Other failing cases or checks: " + excerpt(
            ", ".join(dict.fromkeys(remaining)), 470
        ))
    return "\n".join(lines)


def _line_failure_details(output):
    parts = _PYTEST_SECTION.split(output)
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    summary = sections.get("short test summary info", "")
    nodes = [
        node for status, node in _TEST_RESULT.findall(summary) if status == "FAILED"
    ]
    if not nodes or len(set(nodes)) != len(nodes):
        return {}
    reports = []
    detail = []
    captured = False
    for line in sections.get("FAILURES", "").splitlines():
        if _PYTEST_CRASH_LINE.fullmatch(line):
            location = line[:800]
            message = excerpt("\n".join(detail), 1599 - len(location))
            reports.append(f"{location}\n{message}".rstrip())
            detail = []
            captured = False
        elif line.startswith("-----"):
            captured = True
        elif not captured and (
            line.startswith(("E ", "> ")) or (detail and line.startswith(" "))
        ):
            detail.append(line)
    # Pytest emits both sections in report order. If the terminal output is
    # incomplete or ambiguous, keep the individually named short summaries.
    if len(nodes) != len(reports):
        return {}
    return dict(zip(nodes, reports, strict=True))


@dataclass(frozen=True)
class _PytestRun:
    passed: int
    failed: frozenset[str]
    deselected: int
    named_cases: frozenset[str] | None = None
    assertion_reached: bool = False


def _complete_pytest_run(outcome):
    if not outcome.stdout:
        return None
    return _parse_complete_pytest(outcome.returncode, outcome.stdout.decode("utf-8", "replace"))


def _parse_complete_pytest(returncode, output):
    if returncode not in (0, 1) or not output:
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    lines = text.rstrip().splitlines()
    if not lines or not _PYTEST_TOTALS.fullmatch(lines[-1].strip(" =")):
        return None
    counts = {
        label: int(value)
        for value, label in re.findall(r"(\d+) ([a-z]+)", lines[-1].split(" in ", 1)[0])
    }
    if counts.keys() - {"passed", "failed", "deselected", "warning", "warnings"}:
        return None
    passed, failed = counts.get("passed", 0), counts.get("failed", 0)
    if not passed + failed or bool(failed) != bool(returncode):
        return None
    sections = _PYTEST_SECTION.split(text)
    summaries = [
        body for title, body in zip(sections[1::2], sections[2::2], strict=True)
        if title == "short test summary info"
    ]
    summary = summaries[-1] if summaries else ""
    cases = _TEST_RESULT.findall(summary)
    nodes = frozenset(node for status, node in cases if status == "FAILED")
    if len(cases) != failed or len(nodes) != failed:
        return None
    progress = sections[2] if sections[1:2] == ["test session starts"] else sections[0]
    passed_names = _NAMED_PYTEST_PASS.findall(progress)
    exact_failed = set(_NAMED_PYTEST_FAIL.findall(summary))
    passed_nodes = frozenset(passed_names)
    named_cases = (
        passed_nodes | nodes
        if len(passed_names) == len(passed_nodes) == passed
        and exact_failed == nodes
        and not passed_nodes & nodes
        and not re.search(r"(?m)^!+ stopping after \d+ failures? !+", text)
        else None
    )
    return _PytestRun(passed, nodes, counts.get("deselected", 0), named_cases)


def _lost_passing_test(before, after, path):
    for check, previous in before.items():
        candidate = after.get(check)
        if (
            not previous or previous.named_cases is None
            or not candidate or candidate.named_cases is None
        ):
            continue
        old_passes = previous.named_cases - previous.failed
        lost = old_passes & candidate.failed
        if not candidate.deselected:
            lost |= old_passes - candidate.named_cases
        if case := min((node for node in lost if node.startswith(path + "::")), default=None):
            return case
    return None


def _focused_generated_run(checks, command):
    if command is None:
        return None
    for check in checks:
        observed = tuple(word for word in check.command if not word.startswith("--junitxml="))
        if observed == command:
            return _parse_complete_pytest(check.returncode, check.output)
    return None


def _focused_test_assertions(path, source, run, failures):
    if not run or not run.failed or run.passed or run.deselected or run.named_cases != run.failed:
        return
    try:
        definitions = dict(_symbols(ast.parse(source, filename=path).body))
    except (SyntaxError, ValueError, RecursionError):
        return
    for (_, node), detail in failures.items():
        relative = node.removeprefix("/work/").removeprefix("./")
        if node not in run.failed or not relative.startswith(path + "::"):
            continue
        name = relative.removeprefix(path + "::").split("[", 1)[0].replace("::", ".")
        function = definitions.get(name)
        line = stock_assertion_failure_line(node, detail)
        rewritten = _rewritten_assertion(detail)
        if not rewritten and (match := re.search(
            r"(?m)^E[ \t]+AssertionError:[ \t]+assert[ \t]+(\S[^\n]*)", detail
        )):
            rewritten = match[1]
        if (
            line and rewritten
            and isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(item, ast.Assert) and item.lineno == line for item in ast.walk(function))
        ):
            yield node, line, rewritten


def _focused_test_body_assertion(path, source, run, failures):
    return next(_focused_test_assertions(path, source, run, failures), None) is not None


def _focused_runtime_hint(path, source, run, failures):
    """Identify a runtime failure in the authored case without quoting exception data."""
    if not run or run.passed or run.deselected or len(run.failed) != 1 or run.named_cases != run.failed:
        return ""
    node, = run.failed
    function = node.removeprefix("/work/").removeprefix("./").removeprefix(path + "::")
    if not re.fullmatch(r"test_\w+", function):
        return ""
    method = next((item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef)
                   and item.name == function), None)
    if method is None:
        return ""
    detail = failures.get((0, node), "")
    kind = _named_test_diagnostic_kind(detail)
    if kind not in {"AttributeError", "IndexError", "KeyError", "NameError", "TypeError", "ValueError"}:
        return ""
    rows = source.splitlines()
    for raw in detail.splitlines():
        frame = _PYTEST_GENERIC_FRAME.fullmatch(raw.strip())
        if frame and frame[1].removeprefix("/work/").removeprefix("./") == path:
            row = int(frame[2])
            if method.lineno < row <= method.end_lineno:
                return f"{kind} at authored line {row}: {excerpt(rows[row - 1].strip(), 250)}"
    return ""


def _scalar_assertion(expression):
    """Read a simple rewritten comparison without evaluating executable code."""
    if len(expression) > 160:
        return None
    try:
        comparison = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None
    if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1:
        return None
    operands = []
    for node in (comparison.left, comparison.comparators[0]):
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
            if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
                return None
        if not isinstance(node, ast.Constant) or type(node.value) not in (bool, int, float, type(None)):
            return None
        value = node.value
        if type(value) is float and not isfinite(value):
            return None
        operands.append((type(value), sign * value if type(value) in (int, float) else value))
    return type(comparison.ops[0]), tuple(operands)


def _assertion_comparison_hint(name, fixed, older):
    _, line, expression = fixed
    _, old_line, old_expression = older
    lead = f"Pinned comparison for {name} found failures on both revisions"
    if line != old_line:
        return f"{lead} at fixed test line {line} and ancestor test line {old_line}."
    text = f"{lead} at the same assertion line {line}."
    new, old = _scalar_assertion(expression), _scalar_assertion(old_expression)
    if new and old and new[0] is old[0]:
        differing = [
            side for side, left, right in zip(("left", "right"), new[1], old[1], strict=True)
            if left != right
        ]
        if differing:
            subject = (
                "The left and right scalar operands" if len(differing) == 2
                else f"The {differing[0]} scalar operand"
            )
            return f"{text} {subject} differed."
        return f"{text} The scalar operands matched."
    return text


def _focused_output_reports_pass(output, path):
    sections = _PYTEST_SECTION.split(re.sub(r"\x1b\[[0-9;]*m", "", output))
    progress = sections[2] if sections[1:2] == ["test session starts"] else sections[0]
    return any(
        node.removeprefix("/work/").removeprefix("./").startswith(path + "::")
        for node in _NAMED_PYTEST_PASS.findall(progress)
    )


def _failed_test_line_snapshot(path, draft, diagnostic, run, *, raw=False):
    """Bind the selected failure to a named case from the complete focused run."""
    heading, _, detail = diagnostic.partition("\n")
    if not heading.startswith("Selected failing case: ") or not run or run.named_cases is None:
        return None
    selected = heading.removeprefix("Selected failing case: ")
    if selected not in run.failed or not selected.startswith(path + "::"):
        return None
    if not any(
        row.startswith(("E ", "> ")) or _PYTEST_CRASH_LINE.fullmatch(row)
        for row in detail.splitlines()
    ):
        return None
    function_node = selected.partition("[")[0]
    if raw and (
        run.deselected or selected not in run.named_cases
        or any(node != selected and node.partition("[")[0] == function_node for node in run.named_cases)
    ):
        return None
    qualified = function_node.removeprefix(path + "::").replace("::", ".")
    snapshot = prepare_line_range_edit({path: draft}, (path, qualified))
    max_rows, max_bytes = (90, 6 * 1024) if raw else (26, 2400)
    if (
        snapshot is not None and len(snapshot.lines) <= max_rows
        and sum(len(row.encode()) + 1 for row in snapshot.lines) <= max_bytes
    ):
        return snapshot
    return None


def _test_line_messages(snapshot, note, current, *, repository_usage="", raw=False):
    path, _ = snapshot.symbol
    production = {name: code for name, code in current.items() if name != path}
    failure_source = test_failure_focus(production, note)
    bindings = test_name_error_bindings(path, current[path], note)
    failure_source = (
        TEST_VERIFICATION_SOURCE + excerpt(failure_source, 750) + "\n" if failure_source else ""
    )
    if raw:
        system = (
            _SYSTEM_PREFIX
            + "Return only one complete Python test function with its current name, signature, and decorators.\n"
            + _SYSTEM_SUFFIX
        )
        editor = (
            f"Current failed function {path}::{snapshot.symbol[1]} (quoted source):\n"
            + "\n".join(snapshot.lines)
            + "\nReturn only the complete Python function for this failed test."
        )
    else:
        system = _LINE_RANGE_SYSTEM
        editor = line_range_prompt(snapshot) + LINE_RANGE_ENDING
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                excerpt(note, 1300) + "\n"
                "Repair the failing test setup or call and keep the tested behavior and assertions. "
                + (
                    "Use one Python function, optionally in one fence.\n" if raw else
                    "Use the sample JSON structure with your actual replacement lines.\n"
                )
                + (bindings + "\n" if bindings else "")
                + failure_source
                + (REPOSITORY_PYTEST_USAGE + excerpt(repository_usage, 820) + "\n" if repository_usage else "")
                + editor
            ),
        },
    ]


def _project_generated_seed(path, source, run):
    if not run or not run.named_cases or not run.failed or not run.passed or run.deselected:
        return None
    prefix = path + "::"
    names = {node.removeprefix(prefix) for node in run.named_cases}
    if any(not node.startswith(prefix) for node in run.named_cases) or not all(
        GENERATED_TEST_NAME.fullmatch(name) for name in names
    ):
        return None
    tree = ast.parse(source)
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        return None
    omitted = {node.removeprefix(prefix) for node in run.failed}
    removed = [node for node in definitions if node.name in omitted]
    lines = source.split("\n")
    for node in reversed(removed):
        first = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        del lines[first - 1 : node.end_lineno]
    reduced = "\n".join(lines)
    tree.body = [node for node in tree.body if node not in removed]
    if ast.dump(tree) != ast.dump(ast.parse(reduced)):
        return None
    return GeneratedTestSeed.from_data({
        "source": reduced,
        "passing": sorted(names - omitted),
        "omitted": sorted(omitted),
    })


def _covers_generated_seed(run, path, names):
    if not run or not run.named_cases or run.deselected:
        return False
    passing = run.named_cases - run.failed
    return all(f"{path}::{name}" in passing for name in names)


class _PrivateCheckOutputExceeded(CommandOutputExceeded):
    """A configured check exceeded its output limit on a private proposal."""


def evidence(
    executor,
    task,
    archive,
    budget,
    patch,
    commands,
    *,
    pytest_runs=None,
    static_runs=None,
    unused_import_checks=None,
    on_staged=None,
    private_proposal=False,
    test_check_labels=None,
    check_results=None,
    visible_operator_cases=None,
):
    with executor.verification_runner.workspace(
        archive, budget.remaining, budget.cancelled
    ) as ws:
        tree = executor._stage_baseline(ws, budget)
        if patch:
            executor._git(ws, budget, "apply", "--index", data=patch)
            executor._check_patch_paths(ws, task, budget, tree)
        if on_staged is not None:
            on_staged(ws, tree)
        failures = {}
        for number, command in enumerate(commands):
            budget.claim("command", budget.limits.commands)
            try:
                outcome = executor._command(ws, budget, command, checked=False)
            except CommandOutputExceeded:
                if not private_proposal:
                    raise
                if command in task.pull_checks:
                    ordinal = task.pull_checks.index(command) + 1
                    check = f"Operator acceptance check {ordinal}"
                else:
                    check = f"Required check {task.checks.index(command) + 1}"
                category = check_label(command, ()) or "configured"
                raise _PrivateCheckOutputExceeded(
                    f"{check} ({category}) exceeded the command output limit."
                ) from None
            if pytest_runs is not None and check_label(command, ()) == "pytest":
                pytest_runs[number] = _complete_pytest_run(outcome)
            if check_results is not None or outcome.returncode:
                output = _combined(outcome)
                if check_results is not None:
                    check_results.append(CheckEvidence(command, outcome.returncode, output))
            if outcome.returncode:
                if static_runs is not None and outcome.returncode == 1:
                    locations = ordinary_static_locations(command, output)
                    if locations:
                        static_runs[number] = locations
                        if unused_import_checks is not None and ordinary_unused_import_failure(
                            command, output
                        ):
                            unused_import_checks.add(number)
                nodes = [node for _, node in _TEST_RESULT.findall(output)]
                line_traceback = "--tb=line" in command or any(
                    option == ("--tb", "line") for option in zip(command, command[1:])
                )
                line_details = _line_failure_details(output) if line_traceback else {}
                categories = set()
                if not nodes:
                    detail, kind = _generic_verifier_failure(command, outcome.returncode, output)
                    if kind:
                        categories.add(kind)
                replayed = False
                for node in nodes or [
                    f"check {number + 1} (exit {outcome.returncode})"
                ]:
                    key = (number, node)
                    failures[key] = (
                        detail
                        if not nodes
                        else (line_details.get(node) or _test_detail(output, node))
                    )
                    if (nodes and visible_operator_cases is not None and command in task.pull_checks
                            and not _trusted_operator_node(node, command, visible_operator_cases)):
                        failures[key] = _unpublished_case_detail(output, node, failures[key])
                        if failures[key] == node and not replayed and (replay := _operator_diagnostic_replay(command, node)):
                            replayed = True
                            budget.claim("command", budget.limits.commands)
                            try:
                                repeated = executor._command(ws, budget, replay, checked=False)
                            except CommandOutputExceeded:
                                pass
                            else:
                                report = _combined(repeated)
                                # A rerun only supplies diagnostics. It cannot turn the
                                # configured command's failure into a verification pass.
                                if repeated.returncode == 1 and _TEST_RESULT.findall(report) == [("FAILED", node)]:
                                    failures[key] = _unpublished_case_detail(report, node, _test_detail(report, node))

                    if nodes and test_check_labels is not None and (kind := _named_test_diagnostic_kind(failures[key])):
                        categories.add(kind)
                if test_check_labels is not None:
                    test_check_labels.add(_test_diagnostic_label(command, outcome.returncode, categories))
                if outcome.returncode >= 125 or outcome.returncode < 0:
                    from .worker import ExecutionError

                    raise ExecutionError(
                        "A review verification command could not execute"
                    )
        return failures


def source_excerpt(source, target, max_chars=16000):
    if len(source) <= max_chars:
        return source
    nodes = []
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return source[:max_chars]
    for item in module.body:
        if isinstance(item, ast.ClassDef):
            nodes.extend(
                (f"{item.name}.{getattr(child, 'name', '')}", child)
                for child in item.body
            )
        else:
            nodes.append((getattr(item, "name", ""), item))
    ranked = []
    lines = source.splitlines(True)
    for name, node in nodes:
        start = min(
            [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))]
        )
        chunk = "".join(lines[start - 1 : node.end_lineno])
        rank = 4 * len(words(name) & target) + len(words(chunk) & target)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rank += 30
        ranked.append((rank, start, name, chunk))
    kept = []
    for _, start, name, chunk in sorted(ranked, reverse=True):
        if len(chunk) + 60 <= max_chars:
            kept.append((start, name, chunk))
            max_chars -= len(chunk) + 60
    return "\n".join(
        f"# excerpt {name}, line {n}\n{chunk}" for n, name, chunk in sorted(kept)
    )


def _fixture_excerpt(fixtures, target, *, selected_only=False, selected_first=False):
    selected, helpers = [], []
    source_name, _, selection = target.split("[", 1)[0].partition("::")
    test_name = selection.rsplit("::", 1)[-1]
    matching_paths = [
        path
        for path in fixtures
        if source_name == path or source_name.endswith("/" + path)
    ]
    sources = (
        (fixtures[max(matching_paths, key=len)],)
        if matching_paths
        else fixtures.values()
    )
    for source in sources:
        lines = source.splitlines(True)
        tree = ast.parse(source)
        items = ast.walk(tree) if selected_only else tree.body
        for item in items:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = item.name
            if (selected_only or name.startswith("test_")) and name != test_name:
                continue
            text = "".join(lines[item.lineno - 1 : item.end_lineno])
            (selected if selected_first and name == test_name else helpers).append(text)
    return "".join([*selected, *helpers])[:6000]


def _regression_context(fixtures):
    names = []
    setup = []
    for path, source in fixtures.items():
        try:
            module = ast.parse(source)
        except SyntaxError:
            continue
        names.extend(
            node.name
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
        lines = source.splitlines(keepends=True)
        chunks = []
        for node in module.body:
            if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
            ) or (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("test_")
            ):
                start = min(
                    [
                        node.lineno,
                        *(d.lineno for d in getattr(node, "decorator_list", ())),
                    ]
                )
                chunks.append("".join(lines[start - 1 : node.end_lineno]))
        if chunks:
            setup.append(path + "\n" + excerpt("".join(chunks), 2200 // len(fixtures)))
    return ", ".join(dict.fromkeys(names)), "\n".join(setup)


def _one_test_draft(source):
    try:
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            for node in ast.walk(ast.parse(source))
        ) <= 1
    except (SyntaxError, ValueError, RecursionError):
        return False


def _operator_function_messages(
    projection, review, sources, new_path, ancestor, *, replace_smoke=False, feedback="", prior_function="",
    test_first=False,
):
    requirements = "\n".join(f"- {excerpt(body, 900)}" for _, _, body in review.feedback)
    production = {path: source for path, source in sources.items() if path != new_path}
    focus = test_failure_focus(production, feedback) if feedback else ""
    query = words(" ".join((requirements, projection.preview, ancestor, feedback, prior_function)))
    request = (
        f"Write one original regression test for {new_path}.\n"
        "Return exactly one new Python test function with a test_ name and an ordinary assert. "
        "Define any custom fake inside the function. Use the shown helpers by name. "
        "The runner assembles the referenced helpers and imports. "
        + ("Both the current production checkpoint and the pinned ancestor should fail the requested assertion.\n"
           if test_first else "Assert the requested behavior that passes here and fails on the pinned ancestor.\n")
        + (
            "The current smoke test passed on both revisions. Replace it with a discriminating case.\n"
            if replace_smoke else ""
        )
        + ("\nPrevious authored function and verification feedback (quoted):\n" + feedback if feedback else "")
        + "\nRequested behavior (quoted review):\n"
        + requirements
        + "\n\nReviewed changes (quoted):\n" + excerpt(ancestor, 1400)
        + "\n\nAvailable production APIs:\n"
        + "\n".join(
            _regression_api(path, source, query, 650)[0]
            for path, source in production.items()
        )
        + ("\n\nProduction API at the reported failure:\n" + focus if focus else "")
        + "\n\nApproved setup helpers and imports (quoted):\n" + projection.preview
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PREFIX + "Return only the complete Python test function.\n" + _SYSTEM_SUFFIX},
        {"role": "user", "content": request},
    ]

    if protocol := production_protocol_example(
        projection.preview, production, query,
        review_anchors=tuple((path, line) for path, line, _ in review.feedback),
    ):
        messages[-1]["content"] = request + "\n\nProduction call and response handling (quoted):\n" + protocol
        if wire_size(messages) <= 12 * 1024:
            request = messages[-1]["content"]
        else:
            messages[-1]["content"] = request
    if prior_function:
        for limit in (6000, 3000, 1200, 600):
            messages[-1]["content"] = request + "\n\nPrevious function (quoted):\n" + excerpt(prior_function, limit)
            if wire_size(messages) <= 12 * 1024:
                break
        else:
            messages[-1]["content"] = request
    return messages


def _regression_api(path, content, query, limit, *, anchors=()):
    try:
        module = ast.parse(content)
    except SyntaxError:
        return path, ()
    groups = [("", module.body)]
    groups.extend(
        (node.name + ".", node.body)
        for node in module.body
        if isinstance(node, ast.ClassDef) and (anchors or not node.name.startswith("_"))
    )
    signatures = []
    helpers = []
    for owner, items in groups:
        for node in items:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            private = owner.startswith("_") or (
                node.name.startswith("_") and not (owner and node.name == "__init__")
            )
            if private:
                if node.name.startswith("__") and (owner or node.name.endswith("__")):
                    continue
                first = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
                if not any(first <= anchor <= node.end_lineno for anchor in anchors):
                    continue
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            name = owner + node.name
            signature = f"{prefix} {name}({ast.unparse(node.args)})"
            if node.returns is not None:
                signature += f" -> {ast.unparse(node.returns)}"
            if private:
                helpers.append(signature)
            else:
                signatures.append((len(words(signature) & query), signature))
    signatures.sort(key=lambda item: item[0], reverse=True)
    public = path + "\n" + excerpt("\n".join(text for _, text in signatures), limit)
    return public, helpers


def _test_authoring_messages(
    current, review, fixtures, interfaces, new_path, note, *,
    allow_test_module=False, repository_example="", repository_usage="",
    repair_failed_tests=False, prefer_test_module=False, ancestor_evidence="", extended_context="",
):
    behaviors = list(
        dict.fromkeys(" ".join(body.split()) for _, _, body in review.feedback)
    )
    names, setup = ("", "") if review.test_only else _regression_context(fixtures)
    prioritize_requirements = review.test_only or bool(ancestor_evidence)
    requirements = "\n".join(
        f"- {excerpt(body, 900 if prioritize_requirements else 320)}" for body in behaviors if body
    )
    test_query = words(" ".join([*behaviors, names, setup, note]))
    sources = {path: content for path, content in current.items() if path != new_path}
    failure_source = test_failure_focus(sources, note) if repair_failed_tests else ""
    operator_contract = operator_api_contract(fixtures, sources)
    reviewed_anchors = {}
    for path, line, _ in review.feedback:
        reviewed_anchors.setdefault(path, set()).add(line)
    api = []
    reviewed_helpers = []
    helper_budget = 850
    for path, content in sources.items():
        public, helpers = _regression_api(
            path, content, test_query, 2100 // len(sources), anchors=reviewed_anchors.get(path, ())
        )
        api.append(public)
        for signature in helpers:
            entry = f"{path}\n{signature}\n"
            if (size := len(entry.encode())) <= helper_budget:
                reviewed_helpers.append(entry)
                helper_budget -= size
    system = _TEST_MODULE_SYSTEM if allow_test_module else _SYSTEM
    if allow_test_module and review.test_only and (not current[new_path] or prefer_test_module):
        system = (
            _SYSTEM_PREFIX
            + "Return the complete unpublished pytest module as plain Python. Include all imports, "
            "fixtures and tests needed to run it.\n"
            + _SYSTEM_SUFFIX
        )
    if reviewed_helpers:
        system = system.replace(_SYSTEM_SUFFIX, _TEST_HELPER_SYSTEM_SUFFIX, 1)
    draft = current[new_path]
    bindings = test_name_error_bindings(new_path, draft, note) if repair_failed_tests else ""
    verification_note = f"{bindings}\n{note}" if bindings else note
    definition_guidance = ""
    if allow_test_module and repair_failed_tests and draft:
        definition_guidance = (
            "\nEvery collected case failed. Prefer one complete repaired Python module, "
            "including its imports, fixtures, and test definitions.\n"
            if prefer_test_module else
            f"\nFor a focused repair, prefer one named replacement. Start with REPLACE {new_path}::name "
            "on its own line, substituting the existing function name or Class.method. "
            "Follow with the complete repaired definition, including decorators.\n"
        )
    if len(draft) > 4400:
        position = re.search(rf"{re.escape(new_path)}:([1-9][0-9]*)", note)
        draft = focused_source_excerpt(
            draft,
            note,
            anchor=int(position[1]) if position else None,
            max_chars=4400,
        )
    setup_detail, setup_kind = _pytest_report_diagnostic("", note)
    setup_guidance = (
        f"\nSelected pytest setup failure (quoted data):\n{setup_detail}\n"
        "Pytest fixtures defined only in another test module are unavailable here. "
        "Use self-contained setup in the failing case or a fixture available in this module's scope.\n"
        if repair_failed_tests and setup_kind == "fixture not found" else ""
    )
    scenarios = (
        "Keep the observations of one reviewer-described scenario together in one test execution. "
        "Use separate cases for distinct requested scenarios. "
        if ancestor_evidence else
        "Use two or three focused cases when enough, and cover every distinct requested fixture. "
    )
    request = (
        "Selected failure: new regression tests\n"
        + (TEST_ONLY_REQUIREMENTS + requirements + "\n" if prioritize_requirements else "")
        + (TEST_ANCESTOR_EVIDENCE + ancestor_evidence + "\n" if ancestor_evidence else "")
        + setup_guidance
        + ("\n" if prioritize_requirements else "")
        + f"Edit only {new_path}. "
        + (
            TEST_REPAIR_GUIDANCE
            + " Start with the selected failing case if listed. Keep existing fixtures and coverage where they already work. "
            if repair_failed_tests else
            "Write fresh offline pytest tests for the repository regressions explicitly requested in the review. "
            + scenarios
            + "Keep the module concise. Combine overlapping examples with shared setup or parameters. "
        )
        + "Use independent fixtures. "
        + ("Prefer public APIs. " if reviewed_helpers else "Use public APIs. ")
        + (
            "Use the already-passing acceptance suite as behavior and setup context only. "
            "Avoid copying its cases or reproducing its inventory in the repository tests."
            if not review.test_only else ""
        )
        + (TEST_REVIEW_BEHAVIORS + requirements if not prioritize_requirements else "")
        + (REVIEWED_TEST_HELPERS + "".join(reviewed_helpers) if reviewed_helpers else "")
        + (OPERATOR_API_CONTRACT + operator_contract + "\n" if operator_contract else "")
        + (
            REPOSITORY_PYTEST_EXAMPLE + repository_example + "\n"
            if repository_example else ""
        )
        + (REPOSITORY_PYTEST_USAGE + repository_usage + "\n" if repository_usage else "")
        + (TEST_ONLY_CONTEXT if review.test_only else f"\nOperator test names: {names}\n")
        + (
            TEST_VERIFICATION_FEEDBACK + excerpt(verification_note, 1600) + "\n"
            if note
            else ""
        )
        + (TEST_VERIFICATION_SOURCE + failure_source + "\n" if failure_source else "")
        + "Focused source from verified public APIs:\n"
        + "\n".join(api)
        + "\nFixture for this goal:\n"
        + (f"Unpublished test file {new_path} to revise:\n{draft}\n" if draft else "")
        + ("Operator fixture setup excerpt (quoted data):\n" + setup if not review.test_only else "")
        + "\nImported API declarations:\n"
        + excerpt("\n".join(interfaces), 1300)
        + f"\nNew test file: {new_path}\n"
        + "\nCurrent checkout:\n"
        + (
            EXTENDED_TEST_CONTEXT
            + "Use these examples to understand API calls and setup. Write fresh tests for the review.\n"
            + extended_context + "\n"
            if extended_context else ""
        )
        + f"Edit the test file at the exact path {new_path}.\n"
        + definition_guidance
        + (
            TEST_MODULE_REPLY_ENDING if allow_test_module
            else "Respond with exact SEARCH/REPLACE blocks only."
        )
    )
    return [
        {
            "role": "system",
            "content": system + definition_guidance,
        },
        {"role": "user", "content": request},
    ]


def _draft_review(review, original, current):
    """A GitHub anchor indexes the pinned source, even after a prior goal adds lines."""
    blocks = {}
    feedback = []
    for path, line, body in review.feedback:
        before, after = original[path], current[path]
        if before != after:
            if path not in blocks:
                blocks[path] = difflib.SequenceMatcher(
                    None, before.splitlines(), after.splitlines(), autojunk=False
                ).get_opcodes()
            for _, start, end, new_start, new_end in blocks[path]:
                if start < line <= end:
                    line = max(1, min(new_end, new_start + line - start))
                    break
        feedback.append((path, line, body))
    return replace(review, feedback=tuple(feedback))


def _proposal_messages(
    target,
    current,
    failures,
    needs_tests,
    note,
    review,
    fixtures,
    interfaces,
    new_path,
    *,
    continuing=False,
    named_definitions=False,
    check_command=(),
    rejection="",
    stock_fixture=None,
    preserved_stock_fixture="",
    private_source="",
    mock_detail="",
    parameter_focus="",
    allow_test_module=False,
    repository_example="",
    repository_usage="",
    repair_failed_tests=False,
    prefer_test_module=False,
    ancestor_evidence="",
    extended_context="",
):
    if target == _TEST_GOAL:
        return _test_authoring_messages(
            current,
            review,
            fixtures,
            interfaces,
            new_path,
            note,
            allow_test_module=allow_test_module,
            repository_example=repository_example,
            repository_usage=repository_usage,
            repair_failed_tests=repair_failed_tests,
            prefer_test_module=prefer_test_module,
            ancestor_evidence=ancestor_evidence,
            extended_context=extended_context,
        )
    diagnostic = failures.get(
        target,
        "Create runnable tests covering the requested behavior with offline fixtures.",
    )
    fixture_query = words(
        _fixture_excerpt(fixtures, target[1], selected_only=True)
        if stock_fixture is None
        else stock_fixture
    )
    query = words(target[1] + " " + diagnostic[:600])
    notes = sorted(
        review.feedback,
        key=lambda item: (
            len(words(item[2]) & fixture_query),
            len(words(item[2]) & query),
        ),
        reverse=True,
    )
    path, line, body = notes[0]
    query |= words(body) | fixture_query

    def show_nested_names(path, focus):
        if named_definitions and path == notes[0][0]:
            names = visible_nested_definitions(current[path], focus, line)
            if names:
                focus += "\nExact nested definition headers in the source above:\n" + "\n".join(
                    f"REPLACE {path}::{name}" for name in names
                )
        return focus

    check_focus = (
        (
            "changed definitions invoked by the selected operator case (quoted data)",
            private_source,
        )
        if private_source
        else ("the exercised traversal and its prior bound (quoted)", parameter_focus)
        if parameter_focus
        else focused_check_excerpt(check_command, diagnostic, current)
    )
    if mock_detail:
        diagnostic += "\n" + mock_detail
    if check_focus:
        selected_source, focus = check_focus
        if not (private_source or parameter_focus):
            path = selected_source
        lead = (
            f"Focused source from {selected_source}:\n{show_nested_names(path, focus)}\n"
            f"Failure details:\n{diagnostic}\nReviewer feedback: {body}\n"
        )
    else:
        focus = focused_source_excerpt(
            current[path],
            target[1],
            review_note=body,
            anchor=line,
            max_chars=5900 if rejection else 7600,
        )
        lead = (
            f"Reviewer feedback: {body}\n"
            f"Focused source from {path}:\n{show_nested_names(path, focus)}\n"
            f"Failure details:\n{diagnostic}\n"
        )
    editable = [
        name for name, text in current.items() if text or (needs_tests and not failures)
    ]
    goal = (
        f"Selected failure: {target[1]}\n"
        f"Editable checkout paths (use these exact names): {', '.join(editable)}\n"
        + (rejection + "\n" if rejection else "")
        + lead
    )
    fixture = (
        _fixture_excerpt(fixtures, target[1], selected_first=True)
        if stock_fixture is None
        else stock_fixture
    )
    if diagnostic.startswith(target[1] + "\nObserved pytest failure:") or diagnostic == target[1]:
        fixture = ""
    if preserved_stock_fixture:
        fixture += (
            "\nThis unpublished implementation already passes the following tracked repository case. "
            "Compare its public inputs and expected behavior with the failing review case and preserve both.\n"
            + excerpt(preserved_stock_fixture, 4800)
        )
    request = (
        goal
        + (f"Last proposal feedback for this goal: {note}\n" if note else "")
        + "\nFixture for this goal:\n"
        + fixture
        + "\nImported API declarations:\n"
        + "\n".join(interfaces)
        + (
            ""
            if continuing
            else "\nOther reviewer feedback:\n"
            + "\n".join(item[2] for item in notes[1:])
        )
        + (f"\nNew test file: {new_path}\n" if needs_tests and not failures else "")
        + "\nCurrent checkout:\n"
    )
    ending = (
        f"\nSelected failure: {target[1]}. Fix this failure without breaking passing behavior.\n"
        + (
            DEFINITION_ENDING
            if named_definitions
            else "Respond with exact SEARCH/REPLACE blocks only."
        )
    )

    def with_sources(items):
        source = "\n".join(f"FILE {name}\n{content}" for name, content in items if content)
        return [
            {
                "role": "system",
                "content": _DEFINITION_SYSTEM if named_definitions else _SYSTEM,
            },
            {"role": "user", "content": request + source + ending},
        ]

    # A stock regression can originate outside the historical review anchor.
    # Keep those definitions when the complete request fits its existing budget.
    if (stock_fixture or preserved_stock_fixture) and sum(map(len, current.values())) <= MESSAGE_BYTES:
        complete = with_sources(current.items())
        if wire_size(complete) <= MESSAGE_BYTES:
            return complete
    return with_sources(
        (
            name,
            content
            if name == path and len(content) <= 24000
            else source_excerpt(content, query, 14500 // len(current)),
        )
        for name, content in current.items()
        if content
    )


def _finish_review(executor, task, archive, budget, patch, actor, version):
    result = executor._verify(task, archive, budget, patch, actor=actor)
    return replace(
        result,
        refinement=RefinementEvidence(
            version,
            task.seed,
            task.repository,
            task.instructions,
            {"base": _GUIDANCE},
            (),
            (
                CandidateEvidence(
                    "base", result.patch_digest, "Verified accumulated review fixes"
                ),
            ),
            "base",
        ),
    )


def _coding_completion(
    executor,
    budget,
    messages,
    route,
    planner_input,
    native_source=None,
    advice=None,
    *,
    exclude=(),
    line_snapshot=None,
    scoped_line=False,
    native_test_path=None,
    native_line_retry=None,
    raw_python=False,
    plan_private_method=False,
):
    if plan_private_method:
        messages = private_method_planning_assist(executor, budget, messages, line_snapshot, RETRY_MESSAGE_BYTES)
    fresh_advice = None
    if planner_input is not None:
        fresh_advice = request_coding_advice(executor, budget, messages, *planner_input)
    if fresh_advice is not None:
        _LOG.info("Applying a free planning strategy to the current review goal")
    elif advice is not None:
        _LOG.info("Reusing a free planning strategy for the current review goal")
    fresh_line_snapshot = None
    if (
        line_snapshot is None
        and native_source is None
        and (symbol := localized_retry_symbol(fresh_advice)) is not None
    ):
        numbered = prepare_line_range_edit(planner_input[0], symbol)
        if (
            numbered is not None
            and sum(len(row.encode()) + 1 for row in numbered.lines)
            >= _LONG_LOCALIZED_DEFINITION_BYTES
        ):
            fresh_line_snapshot = line_snapshot = numbered
            route = None
    messages = apply_coding_advice(
        messages,
        advice or fresh_advice,
        reuse=advice is not None,
        line_snapshot=line_snapshot,
    )
    if line_snapshot is not None:
        messages = [{**messages[0], "content": _LINE_RANGE_SYSTEM}, *messages[1:]]
    causal_edit = localized_retry_symbol(advice or fresh_advice) is not None
    if causal_edit:
        messages = [
            {**messages[0], "content": messages[0]["content"] + "\n" + CAUSAL_EDIT_GUIDANCE},
            *messages[1:],
        ]
    if native_source is not None or native_test_path is not None or native_line_retry is not None:
        focused = compact_focused_review(messages, max_bytes=12 * 1024) or messages
        if native_line_retry is not None:
            request = native_line_retry.native_prompt
            tool = (
                selectable_line_tool((native_line_retry.snapshot, *native_line_retry.alternatives))
                if native_line_retry.alternatives else full_method_native_tool(native_line_retry.snapshot)
                if native_line_retry.complete_single_method else NATIVE_LINE_RANGE_TOOL
            )
            instruction = "Call replace_python_line_ranges exactly once with the numbered edits."
        elif native_test_path is not None:
            request = focused[-1]["content"].removesuffix(TEST_MODULE_REPLY_ENDING)
            tool = _TEST_MODULE_TOOL
            instruction = (
                f"Call {_TEST_MODULE_TOOL_NAME} exactly once. Pass the complete Python "
                f"module for {native_test_path} as the code string."
            )
            request += instruction
        else:
            request = focused[-1]["content"].removesuffix(DEFINITION_ENDING)
            tool = NATIVE_DEFINITION_TOOL
            instruction = (
                "Call replace_python_definition exactly once with one complete existing "
                "function or method."
            )
            request += "Use the native definition replacement tool only."
        native_messages = [
            {
                "role": "system",
                "content": _SYSTEM_PREFIX
                + instruction + "\n"
                + (
                    _TEST_HELPER_SYSTEM_SUFFIX
                    if native_test_path is not None
                    and _TEST_HELPER_SYSTEM_SUFFIX in focused[0]["content"]
                    and REVIEWED_TEST_HELPERS in request
                    else _SYSTEM_SUFFIX
                ),
            },
            {
                "role": "user",
                "content": request,
            },
        ]
        try:
            completion = budget.complete(
                executor.gateway,
                "coding",
                native_messages,
                (tool,),
                after_provider=route,
                max_output_tokens=3072 if native_test_path is not None else 2048,
                retry_oversize=False,
                **(
                    {"exclude": exclude}
                    if native_line_retry is not None or native_test_path is not None else {}
                ),
            )
            if native_line_retry is not None:
                candidate = apply_native_line_range_edit(
                    completion.tool_calls, native_line_retry.checkout, native_line_retry.snapshot,
                    allowed_rows=native_line_retry.allowed_rows, alternatives=native_line_retry.alternatives,
                    full_method=native_line_retry.complete_single_method,
                )
            elif native_test_path is None:
                candidate = apply_native_definition(completion.tool_calls, native_source)
            else:
                candidate = _apply_native_test_module(
                    completion.tool_calls, native_test_path, executor.limits.patch_bytes
                )
        except (ProviderExhausted, ValueError):
            # The reserved text request is still usable with providers that cannot
            # return a valid structured edit for this goal.
            pass
        else:
            return completion, candidate, fresh_advice, None
    max_tokens = (
        6144
        if causal_edit or messages[-1]["content"].endswith(DEFINITION_ENDING)
        else 3072
    )
    try:
        completion = budget.complete(
            executor.gateway,
            "coding",
            messages,
            exclude=exclude,
            after_provider=route,
            max_output_tokens=(
                2048 if scoped_line
                else 3072 if fresh_line_snapshot is not None else max_tokens
            ),
            retry_oversize=not (scoped_line or raw_python),
        )
    except ProviderExhausted:
        if fresh_line_snapshot is None:
            raise
        # Keep the verified localization available to the complete-definition
        # retry even when the numbered format has no available coding route.
        completion = None
    return completion, None, fresh_advice, fresh_line_snapshot


def _recovery_digest(task, fixtures, patch):
    context = (
        _GUIDANCE,
        task.repository,
        task.base_sha,
        task.checkpoint_scope,
        task.checks,
        task.pull_checks,
        task.require_new_tests,
        fixtures,
    )
    digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode())
    digest.update(b"\x00")
    digest.update(patch)
    return digest.hexdigest()


def _goal_digest(goal):
    return hashlib.sha256(json.dumps(goal).encode()).hexdigest()


def _rejected_attempt(source, candidate, target, before, after):
    selected = after.get(target)
    name = excerpt(target[1], 180)
    if selected is not None:
        detail = f"{name}: {excerpt(selected, 650)}"
    elif any(key[0] == target[0] for key in after):
        detail = f"No result for {name}; its check reported other failures."
    else:
        detail = f"The check for {name} passed on the rejected candidate."
    new_failures = [
        f"{excerpt(key[1], 180)}: {excerpt(value, 380)}"
        for key, value in after.items()
        if key not in before
    ][:2]
    return (
        "Proposal was discarded. Trace a concrete fixture counterexample through "
        "the rejected changes, then choose a different strategy. Use only the "
        "current checkout text for edits. The rejection below is quote-only evidence.\n"
        "Selected verification on the rejected candidate (quoted data):\n"
        + detail
        + ("\nNew failures:\n" + "\n".join(new_failures) if new_failures else "")
        + "\nRejected edit excerpt (quoted data) against its applied source snapshot:\n"
        + excerpt(patch_for(source, candidate).decode("utf-8"), 1050)
    )


@dataclass(frozen=True)
class _LineRetry:
    advice: object
    snapshot: LineRangeSnapshot
    allowed_rows: frozenset[int] | None = None
    prompt: str = ""
    checkout: dict[str, str] | None = None
    excluded_models: tuple[tuple[str, str], ...] = ()
    native_prompt: str = ""
    native_ready: bool = False
    native_attempted: bool = False
    alternatives: tuple[LineRangeSnapshot, ...] = ()
    checkpoint_method_edit: bool = False
    checkpoint_context: str = ""
    direct_caller_only: bool = False
    caller_routes_reopened: bool = False
    caller_context: str = ""
    checked_caller_attempts: tuple[str, ...] = ()

    @property
    def complete_single_method(self):
        return (
            self.checkpoint_method_edit and not self.alternatives
            and self.allowed_rows == frozenset(range(self.snapshot.first_body_line, len(self.snapshot.lines) + 1))
        )


def _checked_caller_retry(retry, candidate, target, before, after, excluded):
    """Supply only a verifier-observed outcome and the discarded edit to the offered caller."""
    if not retry.direct_caller_only or not retry.caller_context or len(retry.checked_caller_attempts) >= 6:
        return None
    path, method = retry.snapshot.symbol
    if candidate.keys() != retry.checkout.keys() or any(
        candidate[key] != retry.checkout[key] for key in candidate if key != path
    ):
        return None
    trial = prepare_line_range_edit(candidate, retry.snapshot.symbol)
    if trial is None or trial.lines == retry.snapshot.lines:
        return None

    def observed(key):
        detail = after[key]
        marker = "\nObserved pytest failure: "
        if marker not in detail:
            detail = _unpublished_case_detail("", key[1], detail)
        return detail.partition(marker)[2]

    evidence = [
        (index, node, re.sub(r"\b0x[0-9a-fA-F]{6,}\b", "<address>", observed((index, node))))
        for index, node in sorted(after)
    ]
    fingerprint = hashlib.sha256(json.dumps((trial.digest, evidence)).encode()).hexdigest()
    if fingerprint in retry.checked_caller_attempts:
        _LOG.info("Caller-only repair repeated a checked edit and verifier outcome")
        return None
    outcomes = (*retry.checked_caller_attempts, fingerprint)
    if target in after:
        report = f"The selected check still failed: {excerpt(target[1], 180)}."
        if message := observed(target):
            report += "\nObserved pytest failure on this proposal: " + excerpt(message, 520)
    else:
        report = "The selected failure was absent from this run; other verification failed."
    added = sorted(after.keys() - before.keys())
    if added:
        report += f"\nAdditional failing checks: {len(added)}."
        for key in added[:2]:
            report += "\n" + excerpt(key[1], 140)
            if message := observed(key):
                report += " — " + excerpt(message, 180)
    delta = "\n".join(difflib.unified_diff(
        retry.snapshot.lines, trial.lines, fromfile="retained " + method,
        tofile="checked " + method, n=1, lineterm="",
    ))
    visible = frozenset(range(1, len(retry.snapshot.lines) + 1))
    for allowance in (1400, 700, 300):
        context = (
            retry.caller_context + "\nPrevious caller-only proposal was discarded after verification.\n"
            + report + "\nIts local diff (quoted data):\n" + excerpt(delta, allowance)
            + "\nUse this outcome to try a different correction. Edit the retained numbered method below.\n"
        )
        prompt = context + line_range_prompt(retry.snapshot, visible_rows=visible, full_method=True) + _SINGLE_METHOD_ENDING
        native = context + line_range_prompt(retry.snapshot, visible_rows=visible, native=True, full_method=True)
        requests = (
            [{"role": "system", "content": _LINE_RANGE_SYSTEM}, {"role": "user", "content": text}]
            for text in (prompt, native)
        )
        if all(wire_size(request) <= RETRY_MESSAGE_BYTES for request in requests):
            return replace(
                retry, prompt=prompt, native_prompt=native, checked_caller_attempts=outcomes,
                excluded_models=excluded, caller_routes_reopened=False, native_ready=False, native_attempted=False,
            )
    return None


@dataclass(frozen=True)
class _RawTestRetry:
    snapshot: LineRangeSnapshot
    revision: tuple[bytes, bytes]
    excluded_models: tuple[tuple[str, str], ...]


def _review_anchored_line_retry(review, sources, path, node, diagnostic, *, whole_method=False):
    """Number the current reviewed function without requiring operator test source."""
    if len(review.feedback) != 1 or review.feedback[0][0] != path:
        return None
    _, anchor, body = review.feedback[0]
    try:
        tree = ast.parse(sources[path])
    except (SyntaxError, ValueError, RecursionError):
        return None
    owners = [
        name for name, function in _symbols(tree.body)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and min([function.lineno, *(item.lineno for item in function.decorator_list)])
        <= anchor <= function.end_lineno
    ]
    if len(owners) != 1:
        return None
    symbol = path, owners[0]
    snapshot = prepare_line_range_edit(sources, symbol)
    if snapshot is None:
        return None
    heading = (
        f"Selected failure: {node}\n"
        f"Verified pytest result (quoted data):\n{excerpt(diagnostic, 800)}\n"
        f"Reviewed request (quoted data):\n{excerpt(body, 900)}\n"
        "Edit only the shown body rows of the reviewed function. Preserve passing behavior.\n"
    )

    def bounded_retry(snapshot, visible, allowed, *, limit=7500):
        prompt = heading + line_range_prompt(snapshot, visible_rows=visible) + LINE_RANGE_ENDING
        messages = [{"role": "user", "content": prompt}]
        if whole_method:
            messages.insert(0, {"role": "system", "content": _LINE_RANGE_SYSTEM})
        if wire_size(messages) <= limit:
            native_prompt = heading + line_range_prompt(snapshot, visible_rows=visible, native=True)
            return _LineRetry(
                None, snapshot, allowed, prompt, checkout=sources, native_prompt=native_prompt,
                checkpoint_method_edit=whole_method,
            )
        return None

    if whole_method or len(snapshot.lines) <= 90:
        visible = frozenset(range(1, len(snapshot.lines) + 1))
        allowed = frozenset(range(snapshot.first_body_line, len(snapshot.lines) + 1))
        if retry := bounded_retry(snapshot, visible, allowed, limit=RETRY_MESSAGE_BYTES if whole_method else 7500):
            if whole_method:
                # Each alternative is independently bound to the same current file.
                # The original 24-line change limit applies to the selected body.
                choices = []
                choice_instruction = (
                    "Choose exactly one numbered definition below. Reply with that definition's snapshot hash "
                    "and edits to its body only. Row numbers restart for each definition.\n"
                )
                text = retry.prompt.removesuffix(LINE_RANGE_ENDING)
                for name in direct_caller_names(tree, symbol[1]):
                    caller = prepare_line_range_edit(sources, (path, name))
                    if caller is None:
                        continue
                    rows = "\n".join(f"{i} | {row}" for i, row in enumerate(caller.lines, 1))
                    option = (
                        f"\nAlternatively edit direct caller {path}::{name} with snapshot "
                        f'"{caller.digest}". Eligible body rows {caller.first_body_line}–{len(caller.lines)}.\n'
                        f"{rows}\n"
                    )
                    trial = choice_instruction + text + option + LINE_RANGE_ENDING
                    if wire_size([{"role": "system", "content": _LINE_RANGE_SYSTEM}, {"role": "user", "content": trial}]) <= RETRY_MESSAGE_BYTES:
                        choices.append(caller)
                        text += option
                    if len(choices) == 2:
                        break
                if not choices:
                    prompt = heading + line_range_prompt(snapshot, visible_rows=visible, full_method=True) + _SINGLE_METHOD_ENDING
                    native = heading + line_range_prompt(snapshot, visible_rows=visible, native=True, full_method=True)
                    if wire_size([{"role": "system", "content": _LINE_RANGE_SYSTEM}, {"role": "user", "content": prompt}]) <= RETRY_MESSAGE_BYTES:
                        retry = replace(retry, prompt=prompt, native_prompt=native)
                else:
                    retry = replace(
                        retry, prompt=choice_instruction + text + LINE_RANGE_ENDING,
                        native_prompt=choice_instruction + text, alternatives=tuple(choices), checkpoint_context=heading,
                    )
            return retry
    window = prepare_window_line_range_edit(sources, symbol, ((max(1, anchor - 5), anchor + 9),))
    return bounded_retry(*window) if window else None


def _raw_nested_messages(review, source, path, node, diagnostic):
    if len(review.feedback) != 1 or review.feedback[0][0] != path:
        return None
    _, anchor, request = review.feedback[0]
    if not (scope := reviewed_nested_helper(source[path], anchor)):
        return None
    name, helper, calls = scope
    return (path, name), [
        {
            "role": "system",
            "content": _SYSTEM_PREFIX
            + "Return only one complete Python function with the same name and signature.\n"
            + _SYSTEM_SUFFIX,
        },
        {
            "role": "user",
            "content": (
                f"Selected failure: {node}\n"
                f"Reviewed request (quoted):\n{excerpt(request, 900)}\n"
                f"Verified pytest failure (quoted):\n{excerpt(diagnostic, 700)}\n"
                f"Current helper {path}::{name} (quoted):\n{helper}\n"
                f"Calls from its parent (quoted):\n" + "\n".join(calls)
                + "\nReturn only the complete Python function for this helper."
            ),
        },
    ]


@dataclass(frozen=True)
class _StockHedge:
    primary: Future
    due: float
    messages: list[dict[str, str]]
    route: str | None
    exclude: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _RegressionRepair:
    origin: tuple[int, str]
    sources: dict[str, str]
    failures: dict[tuple[int, str], str]
    gained: set[tuple[int, str]]
    introduced: set[tuple[int, str]]
    attempts: int = 0
    stock_origin: bool = False
    private_symbol: tuple[str, str] | None = None
    retries: int = 0
    retry_note: str = ""
    retry_named: bool = False
    retry_target: tuple[int, str] | None = None
    stalled_edits: int = 0
    static_failures: dict[tuple[int, str], int] = field(default_factory=dict)
    stock_collection_id: str | None = None
    dependency_seed: bool = False
    unused_import_checks: frozenset[int] = frozenset()
    excluded_models: tuple[tuple[str, str], ...] = ()

    def next_target(self, commands):
        static = self.static_failures
        if self.dependency_seed and self.introduced:
            static = {key for key in static if key[0] not in self.unused_import_checks}
        if static:
            return min(
                static,
                key=lambda key: (check_label(commands[key[0]], ()) != "mypy", key),
            )
        if (
            self.dependency_seed
            and self.stock_origin
            and self.retries == _STOCK_REPAIR_RETRIES
            and self.stalled_edits == _STOCK_REPAIR_RETRIES
            and self.retry_target in self.introduced
            and len(self.introduced) > 1
        ):
            return min(self.introduced - {self.retry_target})
        return min(self.introduced)


def _private_changed_method(current, tentative):
    changed = [path for path in current if current[path] != tentative[path]]
    if len(changed) != 1:
        return None
    path, = changed
    before, after = current[path].splitlines(), tentative[path].splitlines()
    try:
        original = dict(_symbols(ast.parse(current[path]).body))
        revised = dict(_symbols(ast.parse(tentative[path]).body))
    except (SyntaxError, ValueError, RecursionError):
        return None
    for name, old in original.items():
        new = revised.get(name)
        if not isinstance(old, (ast.FunctionDef, ast.AsyncFunctionDef)) or type(new) is not type(old):
            continue
        if (old.lineno == new.lineno and before[:old.lineno] == after[:new.lineno]
                and before[old.end_lineno:] == after[new.end_lineno:]):
            return path, name
    return None


def _private_method_retry(current, repair, target, review):
    snapshot = prepare_line_range_edit(repair.sources, repair.private_symbol)
    if snapshot is None:
        return None
    accepted = prepare_line_range_edit(current, repair.private_symbol)
    if accepted is None:
        return None
    visible = frozenset(range(1, len(snapshot.lines) + 1))
    allowed = frozenset(range(snapshot.first_body_line, len(snapshot.lines) + 1))
    observed = repair.failures[target].partition("\nObserved pytest failure: ")[2]
    context = (
        "Private regression repair. The numbered tentative method fixed "
        + ", ".join(excerpt(node, 120) for _, node in sorted(repair.gained))
        + ". Preserve that gain while correcting " + excerpt(target[1], 160) + ".\n"
        + "Requested behavior (quoted): " + excerpt(" ".join(body for _, _, body in review.feedback), 600) + "\n"
        + ("Observed pytest failure: " + excerpt(observed, 450) + "\n" if observed else "")
        + (excerpt(repair.retry_note, 450) + "\n" if repair.retry_note else "")
    )
    delta = "\n".join(difflib.unified_diff(
        accepted.lines, snapshot.lines, fromfile="accepted method", tofile="tentative method", n=1, lineterm="",
    ))
    numbered = line_range_prompt(snapshot, visible_rows=visible, full_method=True) + _SINGLE_METHOD_ENDING
    for allowance in (1800, 900, 400, 0):
        evidence = "\nVerified change that fixed the target and broke the control (quoted diff):\n" + excerpt(delta, allowance) + "\n" if allowance else ""
        prompt = context + evidence + numbered
        if wire_size([{"role": "system", "content": _LINE_RANGE_SYSTEM}, {"role": "user", "content": prompt}]) <= RETRY_MESSAGE_BYTES:
            return _LineRetry(None, snapshot, allowed_rows=allowed, prompt=prompt, checkout=repair.sources,
                              checkpoint_method_edit=True)
    return None


def _private_case_outcome(goal, after, runs, commands):
    number, node = goal
    if goal in after:
        return "failed"
    run = runs.get(number)
    if run and not run.deselected and run.passed:
        if run.named_cases is not None and node in run.named_cases - run.failed:
            return "passed"
        # Quiet pytest may suppress individual passing IDs. A single explicit
        # selector and one passing case still identify the original test.
        selected = [arg for arg in commands[number] if "::" in arg]
        if (run.passed == 1 and not run.failed and len(selected) == 1
                and selected[0].partition("::")[2] == node.partition("::")[2]
                and PurePosixPath(selected[0].partition("::")[0]).name == PurePosixPath(node.partition("::")[0]).name):
            return "passed"
    return "unconfirmed"


def _retry_private_method(repair, target, editor, note, *, checked=False):
    if not repair.private_symbol or repair.retries >= 2 or not editor.configured_model:
        return None
    model = editor.route, editor.configured_model
    if model in repair.excluded_models:
        return None
    return replace(repair, retries=repair.retries + 1, retry_target=target, retry_note=note,
                   stalled_edits=repair.stalled_edits + int(checked), excluded_models=(*repair.excluded_models, model))


def _retry_stock_repair(repair, target, note, named_definitions, *, checked=False):
    if repair.retries >= _STOCK_REPAIR_RETRIES:
        return None
    streak = repair.stalled_edits + 1 if repair.retry_target == target else 1
    return replace(
        repair,
        retries=repair.retries + 1,
        retry_note=note,
        retry_named=named_definitions,
        retry_target=target,
        stalled_edits=streak if checked else 0,
    )


def _localized_static_failures(keys, static_runs, sources, allowed):
    if len(keys) > _STOCK_REPAIR_STATIC_LIMIT:
        return None
    line_counts = {path: len(sources[path].splitlines()) for path in allowed}
    checked = {}
    for number, node in keys:
        locations = static_runs.get(number)
        if not locations or node != f"check {number + 1} (exit 1)":
            return None
        for path, row in locations:
            path = path.removeprefix("/work/").removeprefix("./")
            if not 1 <= row <= line_counts.get(path, 0):
                return None
        checked[number, node] = len(locations)
    return checked


def _named_operator_cases(keys, commands):
    # A missing node is meaningful only when pytest ran the complete check.
    incomplete = {
        "-x",
        "--exitfirst",
        "--lf",
        "--last-failed",
        "--sw",
        "--stepwise",
        "--maxfail",
    }
    return all(
        "::" in node
        and check_label(commands[number], ()) == "pytest"
        and not any(arg.split("=", 1)[0] in incomplete for arg in commands[number])
        for number, node in keys
    )


def _operator_case_names(fixtures):
    names = {}
    for path, source in fixtures.items():
        try:
            body = ast.parse(source).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        names[f"{review_recovery.OPERATOR_CHECK_ROOT}/{path}"] = {
            name.replace(".", "::")
            for name, node in _symbols(body)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test")
        }
    return names


def _trusted_operator_node(node, command, names):
    path, _, selection = node.partition("::")
    if not selection:
        return False
    original_path = path
    while path.startswith("../"):
        path = path[3:]
    path = path.removeprefix("./")
    plain_selection = selection.split("[", 1)[0]
    matches = 0
    for argument in command:
        configured, _, selected = argument.partition("::")
        if plain_selection not in names.get(configured, ()):
            continue
        if original_path != configured and (
            path.startswith("/") or not configured.endswith("/" + path)
        ):
            continue
        if not selected or selection == selected or selection.startswith(
            (selected + "::", selected + "[")
        ):
            matches += 1
    return matches == 1


def _selected_operator_fixture(node, command, names, fixtures):
    selectors = [arg for arg in command if _trusted_operator_node(node, (arg,), names)]
    if len(selectors) != 1:
        return None
    configured, _, _ = selectors[0].partition("::")
    selection = node.partition("::")[2]
    relative = configured.removeprefix(review_recovery.OPERATOR_CHECK_ROOT + "/")
    source = fixtures.get(relative)
    if not source or not selection:
        return None
    qualified = selection.partition("[")[0].replace("::", ".")
    try:
        declarations = [
            item for name, item in _symbols(ast.parse(source).body) if name == qualified
        ]
    except (SyntaxError, ValueError, RecursionError):
        return None
    if len(declarations) != 1 or not isinstance(
        declaration := declarations[0], ast.FunctionDef
    ):
        return None
    fixture = dedent(
        "".join(
            source.splitlines(keepends=True)[
                declaration.lineno - 1 : declaration.end_lineno
            ]
        )
    )
    return (
        (f"{configured}::{selection}", fixture)
        if 0 < len(fixture.encode()) <= 6000
        else None
    )


def _same_pytest_selection(before, after):
    return (
        before is not None
        and after is not None
        and before.passed + len(before.failed) == after.passed + len(after.failed)
        and before.deselected == after.deselected
    )


def _preserved_stock_gain(origin, baseline, candidate, failures):
    number, node = origin
    before, after = baseline.get(number), candidate.get(number)
    return (
        _same_pytest_selection(before, after)
        and node in before.failed
        and not after.failed
        and all(failed_check != number for failed_check, _ in failures)
    )


def _verified_operator_runs(before, after, failures, operator_commands, names):
    reported = set()
    for number, command in enumerate(operator_commands):
        if check_label(command, ()) != "pytest":
            continue
        baseline, candidate = before.get(number), after.get(number)
        if not _same_pytest_selection(baseline, candidate) or baseline.failed:
            return False
        for node in candidate.failed:
            if not _trusted_operator_node(node, command, names):
                return False
            reported.add((number, node))
    return reported == failures


def _observed_operator_failures(runs, failures, operator_commands, names):
    reported = set()
    for number, command in enumerate(operator_commands):
        if check_label(command, ()) != "pytest":
            continue
        run = runs.get(number)
        if run is None:
            return False
        for node in run.failed:
            if not _trusted_operator_node(node, command, names):
                return False
            reported.add((number, node))
    return reported == failures


def _qualified_stock_operator_cases(
    before, after, failures, commands, names, *, dependency_seed=False
):
    size = len(failures)
    limit = _DEPENDENCY_SEED_CASE_LIMIT if dependency_seed else _STOCK_REPAIR_CASE_LIMIT
    if size > limit or not _verified_operator_runs(
        before, after, failures, commands, names
    ):
        return False
    if not dependency_seed:
        return True
    passed = sum(
        after[number].passed
        for number, command in enumerate(commands)
        if check_label(command, ()) == "pytest"
    )
    return passed >= size


class _GuardBudget:
    """Charge the normal command budget while capping optional verification time."""

    def __init__(self, budget, *, wall_seconds=None, reserve=25):
        self.budget = budget
        self.cancelled = budget.cancelled
        self.limits = budget.limits
        # Leave room for the normal private repair if neither guarded edit verifies.
        limit = _GUARD_SECONDS if wall_seconds is None else wall_seconds
        self.deadline = time.monotonic() + min(limit, max(0, budget.remaining() - reserve))

    def remaining(self):
        remaining = min(self.budget.remaining(), self.deadline - time.monotonic())
        if remaining <= 0:
            raise LimitExceeded("Guarded review verification timed out")
        return remaining

    def claim(self, kind, limit):
        self.remaining()
        return self.budget.claim(kind, limit)


def _pinned_restore_budget(budget, *, dependency_seed):
    return _GuardBudget(
        budget,
        wall_seconds=_DEPENDENCY_SEED_PROBE_SECONDS if dependency_seed else None,
        reserve=60,
    )


class _InteractiveBudget(_GuardBudget):
    def __init__(self, budget, turns, tools):
        super().__init__(budget, wall_seconds=420, reserve=_INTERACTIVE_RESERVE_SECONDS)
        self.limits = replace(budget.limits, rollout_requests=turns)
        self.turns_left = turns
        self.tools_left = tools

    def model_requests_remaining(self):
        return min(self.turns_left, self.budget.model_requests_remaining())

    def complete(self, *args, **kwargs):
        if not self.model_requests_remaining():
            raise _RequestBudgetExceeded("Interactive review model request limit reached")
        available = self.remaining() - _INTERACTIVE_CAPTURE_RESERVE_SECONDS
        if available <= 0:
            raise LimitExceeded("Interactive review ran out of time for another model request")
        self.turns_left -= 1
        kwargs.update(
            timeout_cap=min(120, available), retry_oversize=False, max_output_tokens=3072
        )
        return self.budget.complete(*args, **kwargs)

    def claim(self, kind, limit):
        if not self.tools_left:
            raise _RequestBudgetExceeded("Interactive review tool limit reached")
        self.tools_left -= 1
        return super().claim(kind, limit)

    def reject(self, *args, **kwargs):
        return self.budget.reject(*args, **kwargs)


def _private_repair_context(current, repair):
    if repair.stock_origin:
        gained = "the entire repository pytest check containing " + excerpt(repair.origin[1], 160)
    else:
        gained = ", ".join(excerpt(node, 160) for _, node in sorted(repair.gained)[:4])
    return (
        "Private regression repair. Fix the selected newly failing case. "
        "Preserve all previously passing cases and the improvement in: "
        + gained
        + ". Edit the current tentative source. Use this diff only as evidence.\n"
        + "Unpublished changes relative to the accepted checkout (quoted data):\n"
        + excerpt(patch_for(current, repair.sources).decode("utf-8"), 1050)
    )


def solve(executor, task, archive, budget, version):
    review = task.review_recovery
    pinned_ancestor = bool(task.regression_base and task.regression_archive)
    production_regression = bool(
        pinned_ancestor and task.regression_edit_path and task.pull_checks
        and task.inline_review_threads == 1
    )
    if not (
        review
        and (
            review.test_only or production_regression
            or (task.pull_checks and task.inline_review_threads >= 2)
        )
        and task.save_checkpoint
    ):
        return None
    paths = list(dict.fromkeys(path for path, _, _ in review.feedback))
    if not paths or len(paths) > 3 or any(not path.endswith(".py") for path in paths):
        return None
    new_path = "tests/test_review_regressions.py"
    if production_regression and (review.test_only or paths != [task.regression_edit_path]):
        return None
    if (review.test_only or production_regression) and not (
        task.inline_review_threads == 1 and task.require_new_tests and task.verify_base_first
        and task.test_update is None and focused_pytest(task.checks, (new_path,))
    ):
        return None
    with executor.runner.workspace(archive, budget.remaining, budget.cancelled) as ws:
        tree = executor._stage_baseline(ws, budget)
        context = operator_acceptance_context(
            task.pull_checks,
            lambda argv, cap: executor._command(
                ws, budget, argv, cap=cap, checked=False
            ),
        )
        if not context and not (review.test_only or production_regression):
            return None
        fixtures = json.loads(context.split("\n", 1)[1]) if context else {}
        original = {
            path: executor._read_recovery_source(ws, budget, task, tree, path)
            for path in paths
        }
        if any(
            not value or len(value) > 128000 or not value.endswith("\n")
            for value in original.values()
        ):
            return None
        if task.require_new_tests:
            if review.test_only or production_regression:
                candidates = (
                    new_path,
                    *(f"tests/test_review_regressions_{index}.py" for index in range(2, 33)),
                )
                occupied = set(executor._git(
                    ws, budget, "--literal-pathspecs", "ls-tree", "-z", "--name-only",
                    tree, "--", *candidates,
                ).stdout.split(b"\0"))
                new_path = next((
                    path for path in candidates
                    if path.encode() not in occupied and not _protected(path, task, new_test=True)
                ), None)
                if new_path is None:
                    return None
            elif (
                _protected(new_path, task, new_test=True)
                or executor._git(ws, budget, "ls-tree", tree, "--", new_path).stdout
            ):
                return None
            original[new_path] = ""
        current = original.copy()
        draft = None if review.test_only else task.checkpoints.get(_CHECKPOINT)
        actor = None
        restored_checkpoint = bool(draft and draft.guidance == _GUIDANCE and draft.base_sha == task.base_sha)
        if restored_checkpoint:
            executor._git(ws, budget, "apply", "--index", data=draft.patch)
            executor._check_patch_paths(ws, task, budget, tree)
            for path in current:
                blob = executor._git(
                    ws,
                    budget,
                    "cat-file",
                    "blob",
                    ":" + path,
                    checked=False,
                    cap=256 * 1024,
                )
                if blob.returncode == 0:
                    current[path] = blob.stdout.decode()
            if draft.last_editor:
                actor = RouteEvidence(
                    "coding",
                    draft.last_editor[0],
                    draft.last_editor[1],
                    {},
                    draft.last_editor[1],
                )
        # Include repository-defined exception and API names that the reviewed module imports.
        interfaces = []
        for source in original.values():
            try:
                module = ast.parse(source)
            except (SyntaxError, ValueError, RecursionError):
                continue
            for node in module.body:
                if (
                    not isinstance(node, ast.ImportFrom)
                    or not node.module
                    or node.level
                ):
                    continue
                local = node.module.replace(".", "/") + ".py"
                if local in paths or len(interfaces) >= 3:
                    continue
                companion = executor._read_recovery_source(
                    ws, budget, task, tree, local
                )
                if companion:
                    outline = [
                        line
                        for line in companion.splitlines()
                        if line.startswith(("class ", "def ", "    def "))
                    ]
                    interfaces.append(local + "\n" + "\n".join(outline)[:2200])
    commands = tuple(dict.fromkeys((*task.pull_checks, *task.checks)))
    assertions = operator_assertions(fixtures, review_recovery.OPERATOR_CHECK_ROOT)
    operator_cases = _operator_case_names(fixtures)
    review_evidence = functools.partial(evidence, visible_operator_cases=operator_cases)
    patch = patch_for(original, current)
    pytest_runs = {}
    failures = review_evidence(executor, task, archive, budget, patch, commands, pytest_runs=pytest_runs)
    resumed_verified_checkout = (
        current if (
            production_regression and restored_checkpoint and draft.key == _CHECKPOINT
            and patch and patch == draft.patch and not failures and not current[new_path]
        ) else None
    )
    resumed_operator_checkout = (
        current if production_regression and restored_checkpoint and draft.key == _CHECKPOINT
        and patch and patch == draft.patch and len(failures) == 1 and not current[new_path]
        else None
    )
    if review.test_only and failures:
        raise ExecutionError("Required checks failed on the unchanged baseline for the test-only review")
    notes = {}
    rejected = {}
    last_routes = {}
    # None marks a goal and text format whose alternate routes were exhausted.
    model_failures = {}
    attempts = {}
    malformed = {}
    native_ready = set()
    native_attempted = set()
    no_progress = {}
    retained_failures = failures
    retained_identity = _recovery_digest(task, fixtures, patch)
    retained_recovery = task.solver_recovery
    alternate = None
    test_seed = None
    durable_witness = None
    tradeoff = None
    def witness_identity():
        return hashlib.sha256(json.dumps((
            retained_identity, task.regression_base, new_path, review.feedback,
        )).encode()).hexdigest()

    restored_raw_targets = set()
    restored_raw_patch = patch
    if retained_recovery and retained_recovery.identity == retained_identity:
        alternate = retained_recovery.alternate
        tradeoff = retained_recovery.tradeoff
        saved_witness = retained_recovery.red_witness
        if (current is resumed_operator_checkout and saved_witness is not None
                and saved_witness.identity == witness_identity()):
            durable_witness = saved_witness
        if task.require_new_tests and not failures and not current[new_path]:
            test_seed = retained_recovery.test_seed
        goals = {goal: (bad, stalled) for goal, bad, stalled in retained_recovery.goals}
        for target in failures:
            bad, stalled = goals.get(_goal_digest(target), (0, 0))
            if production_regression and min(bad, stalled) >= RECOVERY_MODE_THRESHOLD:
                restored_raw_targets.add(target)
            if bad:
                malformed[target] = bad
            if stalled:
                no_progress[target] = stalled

    def retain_recovery(durable=False):
        nonlocal retained_recovery
        if task.save_solver_recovery is None:
            return
        goals = [
            (_goal_digest(target), malformed.get(target, 0), no_progress.get(target, 0))
            for target in retained_failures
            if malformed.get(target) or no_progress.get(target)
        ]
        goals.sort(key=lambda goal: (-max(goal[1:]), goal[0]))
        state = (
            SolverRecovery(retained_identity, tuple(goals[:RECOVERY_GOAL_LIMIT]), alternate, test_seed, durable_witness, tradeoff)
            if goals or alternate is not None or test_seed is not None or durable_witness is not None or tradeoff is not None
            else None
        )
        if state != retained_recovery:
            task.save_solver_recovery(state, durable)
            retained_recovery = state

    def count_recovery(counter, target):
        previous = counter.get(target, 0)
        next_count = min(RECOVERY_MODE_THRESHOLD, previous + 1)
        counter[target] = next_count
        if previous != next_count and next_count == RECOVERY_MODE_THRESHOLD:
            # Persist mode changes before the worker can lose its lease. Shorter
            # attempts and verified checkpoint writes coalesce other updates.
            retain_recovery(durable=True)

    def count_model_failure(rotation_key, snapshot, editor):
        counts = model_failures.get(rotation_key)
        if counts is not None and snapshot is current and editor.configured_model:
            counts[editor.route, editor.configured_model] += 1

    def is_isolated_stock_failure(target, checked, runs):
        return (
            failures.keys() == checked.keys() == {target}
            and stock_fixtures.get((target, failures[target]))
            and _same_pytest_selection(pytest_runs.get(target[0]), runs.get(target[0]))
            and pytest_runs[target[0]].failed == runs[target[0]].failed == {target[1]}
            and _verified_operator_runs(
                pytest_runs, runs, set(), task.pull_checks, operator_cases
            )
        )

    def tracked_assertion_note(target, sources, checked):
        if not _named_operator_cases({target}, commands):
            return ""
        changed_paths = {
            path
            for path in current.keys() | sources.keys()
            if current.get(path) != original.get(path)
            or sources.get(path) != original.get(path)
        }
        return stock_assertion_progress(
            archive,
            target[1],
            failures[target],
            checked[target],
            changed_paths=changed_paths,
        )

    def qualified_stock_gain(
        target, checked, runs, operator_failures, static_failures, *, dependency_seed=False
    ):
        return (
            stock_fixtures.get((target, failures.get(target)))
            and _named_operator_cases({target}, commands)
            and _preserved_stock_gain(target, pytest_runs, runs, checked)
            and (
                static_failures
                or (operator_failures and (checked.keys() - operator_failures) < failures.keys())
            )
            and _qualified_stock_operator_cases(
                pytest_runs, runs, operator_failures, task.pull_checks, operator_cases,
                dependency_seed=dependency_seed,
            )
        )

    def private_retry_feedback(repair, target, after, runs, static_runs):
        novel = None
        new_static = ""
        for key, diagnostic in after.items():
            if key not in repair.failures:
                novel = key
                break
            if key not in repair.static_failures or key[0] not in static_runs:
                continue
            command = commands[key[0]]
            before = repair.failures[key]
            if len(
                ordinary_static_locations(command, before)
            ) != repair.static_failures[key] or len(
                ordinary_static_locations(command, diagnostic)
            ) != len(static_runs[key[0]]):
                continue
            new_static = static_error_message(command, diagnostic, before=before)
            if new_static:
                novel = key
                break
        key = novel or target
        if key not in after:
            return ""
        number, node = key
        command, diagnostic = commands[number], after[key]
        tool = check_label(command, ())
        scope = "review" if number < len(task.pull_checks) else "repository"
        title = f"{scope} check {number + 1}"
        detail = fixture = ""
        run = runs.get(number)
        review_case = scope == "review" and _trusted_operator_node(
            node, command, operator_cases
        )
        repository_case = scope == "repository" and _named_operator_cases(
            {key}, commands
        )
        if (review_case or repository_case) and run is not None and node in run.failed:
            path, _, selection = node.partition("::")
            path = path.removeprefix("/work/").removeprefix("./")
            if review_case or path.startswith(("/", "../")):
                path = path.rsplit("/", 1)[-1]
            title = f"{scope} pytest case {excerpt(f'{path}::{selection}', 170)}"
            errors = [
                line[2:].strip()
                for line in diagnostic.splitlines()
                if line.startswith("E ")
            ]
            detail = "\n".join(errors[:2])
            if not detail:
                detail = next(
                    (
                        line.rsplit(": ", 1)[-1]
                        for line in diagnostic.splitlines()
                        if _PYTEST_CRASH_LINE.fullmatch(line)
                    ),
                    "",
                )
            if novel and review_case:
                fixture = _fixture_excerpt(fixtures, node, selected_only=True)
            elif novel and repository_case:
                fixture = stock_pytest_excerpt(archive, command, node, diagnostic)
        elif tool in {"mypy", "ruff"}:
            title = f"{tool} check {number + 1}"
            if new_static:
                detail = new_static
            elif number in static_runs:
                detail = (
                    check_details(command, diagnostic)
                    + " "
                    + static_error_message(command, diagnostic)
                )
        heading = (
            "New failure from the rejected edit"
            if novel
            else "Selected case on the rejected edit"
        )
        return (
            f"\n{heading} (quoted verification): {title}"
            + (f"\n{excerpt(detail, 260)}" if detail else "")
            + (
                f"\nPreserved fixture (quoted, read-only):\n{excerpt(fixture, 420)}"
                if fixture
                else ""
            )
        )

    def identify_stock_collection(target, draft_patch, probe_budget, *, failed=False):
        run = pytest_runs[target[0]]
        return stock_collection(
            executor,
            task,
            archive,
            probe_budget,
            draft_patch,
            commands[target[0]],
            target[1],
            run.passed + len(run.failed),
            failed=failed,
        )

    def private_identity():
        return _goal_digest(
            (retained_identity, review.feedback, task.protected_paths, task.test_update)
        )

    def tradeoff_identity():
        return _goal_digest((private_identity(), task.regression_base, new_path))

    def checked_tradeoff(candidate_patch, checked, runs, probe):
        if (current is not resumed_operator_checkout or len(failures) != 1 or len(checked) != 1
                or failures.keys() == checked.keys()):
            return None
        origin, = failures
        control, = checked
        if (any(not 0 <= key[0] < len(task.pull_checks) for key in (origin, control))
                or not _named_operator_cases({origin, control}, commands)):
            return None
        for number, command in enumerate(commands):
            if check_label(command, ()) != "pytest":
                continue
            first, second = pytest_runs.get(number), runs.get(number)
            if (not _same_pytest_selection(first, second) or first.deselected
                    or (first.named_cases is not None and second.named_cases is not None
                        and second.named_cases != first.named_cases)):
                return None
        if origin[1] not in pytest_runs[origin[0]].failed or control[1] not in runs[control[0]].failed:
            return None
        for case, revision, diagnostics, observed in (
            (origin, candidate_patch, checked, runs), (control, patch, failures, pytest_runs),
        ):
            if _private_case_outcome(case, diagnostics, observed, commands) == "passed":
                continue
            focused = _operator_diagnostic_replay(commands[case[0]], case[1])
            if not focused:
                return None
            proof = {}
            failed = review_evidence(executor, task, archive, probe, revision, (focused,), pytest_runs=proof)
            run = proof.get(0)
            if failed or not run or run.passed != 1 or run.failed or run.deselected:
                return None
        return origin, control

    def restore_tradeoff():
        if (tradeoff.identity != tradeoff_identity() or current is not resumed_operator_checkout
                or executor.runner is executor.verification_runner or not task.save_solver_recovery
                or len(tradeoff.patch) > executor.limits.patch_bytes):
            return None
        sources = original.copy()
        scope = {path: original[path] for path in paths if not _protected(path, task, new_test=False)}
        probe = _GuardBudget(budget, wall_seconds=180, reserve=60)

        def capture(ws, tree):
            sources.update(executor._capture_scoped_sources(ws, probe, tree, scope, text_only=True))
            if patch_for(original, sources) != tradeoff.patch:
                raise ValueError("Tradeoff modifies files outside the reviewed source")

        runs = {}
        try:
            checked = review_evidence(executor, task, archive, probe, tradeoff.patch, commands,
                                      pytest_runs=runs, on_staged=capture)
            cases = checked_tradeoff(tradeoff.patch, checked, runs, probe)
            symbol = _private_changed_method(current, sources) if cases else None
            if symbol and tuple(map(_goal_digest, cases)) == (tradeoff.origin, tradeoff.control):
                origin, control = cases
                _LOG.info("Reverified a private tradeoff against the accepted checkpoint and all guards")
                return _RegressionRepair(origin, sources, checked, {origin}, {control}, private_symbol=symbol)
        except Cancelled:
            raise
        except (ExecutionError, OSError, ValueError):
            budget.remaining()
        return None

    def restore_private_progress():
        nonlocal alternate
        if alternate is None or task.save_solver_recovery is None:
            return None
        if (
            not budget.model_requests_remaining()
            or budget.commands_remaining() < 2 * len(commands) + 3
            or budget.remaining() < 90
        ):
            return None
        saved, alternate = alternate, None
        if (
            saved.identity != private_identity()
            or len(failures) != 1
            or not patch
            or len(saved.patch) > executor.limits.patch_bytes
            or len(saved.preceding_cases)
            > (
                _DEPENDENCY_SEED_CASE_LIMIT
                if saved.dependency_seed
                else _STOCK_REPAIR_CASE_LIMIT
            )
        ):
            return None
        target = next(iter(failures))
        baseline_run = pytest_runs.get(target[0])
        if (
            saved.origin != _goal_digest(target)
            or not _named_operator_cases({target}, commands)
            or baseline_run is None
            or baseline_run.failed != {target[1]}
        ):
            return None
        fixture = stock_pytest_excerpt(
            archive, commands[target[0]], target[1], failures[target]
        )
        if not fixture:
            return None
        stock_fixtures[target, failures[target]] = fixture
        allowed = {path for path in paths if not _protected(path, task, new_test=False)}
        sources = original.copy()
        probe = _GuardBudget(budget, wall_seconds=180, reserve=60)

        def capture(ws, tree):
            scope = {path: original[path] for path in allowed}
            updates = executor._capture_scoped_sources(
                ws, probe, tree, scope, text_only=True
            )
            sources.update(updates)
            if not updates or patch_for(original, sources) != saved.patch:
                raise ValueError("Invalid private review patch")

        runs, static_runs = {}, {}
        unused_import_checks = set()
        try:
            checked = review_evidence(
                executor,
                task,
                archive,
                probe,
                saved.patch,
                commands,
                pytest_runs=runs,
                static_runs=static_runs,
                unused_import_checks=unused_import_checks
                if saved.dependency_seed
                else None,
                on_staged=capture,
            )
            operator = {key for key in checked if key[0] < len(task.pull_checks)}
            static = _localized_static_failures(
                checked.keys() - operator, static_runs, sources, allowed
            )
            if (
                {_goal_digest(key) for key in operator} != set(saved.cases)
                or static is None
                or {_goal_digest(key): count for key, count in static.items()}
                != dict(saved.static)
                or not qualified_stock_gain(
                    target,
                    checked,
                    runs,
                    operator,
                    static,
                    dependency_seed=saved.dependency_seed,
                )
                or identify_stock_collection(target, patch, probe, failed=True)
                != saved.collection
                or identify_stock_collection(target, saved.patch, probe)
                != saved.collection
            ):
                return None
        except Cancelled:
            raise
        except LimitExceeded:
            budget.remaining()
            alternate = saved
            return None
        except (ExecutionError, ValueError):
            budget.remaining()
            return None
        alternate = saved
        _LOG.info(
            "Resumed independently verified private stock progress with %d review cases",
            len(operator),
        )
        return _RegressionRepair(
            target,
            sources,
            checked,
            set(),
            operator,
            stock_origin=True,
            static_failures=static,
            stock_collection_id=saved.collection,
            dependency_seed=saved.dependency_seed,
            unused_import_checks=frozenset(unused_import_checks),
        )

    def save_private_progress(repair, successor, proposed, editor):
        nonlocal alternate
        if (
            task.save_solver_recovery is None
            or not repair.stock_collection_id
            or not (editor.role == "coding" and editor.configured_model)
            or not successor.introduced
            or not successor.introduced < repair.introduced
            or not 0 < len(proposed) <= PRIVATE_PATCH_BYTES
            or len(repair.introduced) > RECOVERY_GOAL_LIMIT
            or any(count > 2048 for count in successor.static_failures.values())
            or any(
                source != original.get(path)
                for path, source in successor.sources.items()
                if path not in paths or _protected(path, task, new_test=False)
            )
        ):
            return
        state = PrivateRecovery(
            private_identity(),
            _goal_digest(repair.origin),
            repair.stock_collection_id,
            proposed,
            tuple(sorted(map(_goal_digest, repair.introduced))),
            tuple(sorted(map(_goal_digest, successor.introduced))),
            tuple(
                sorted(
                    (_goal_digest(key), count)
                    for key, count in successor.static_failures.items()
                )
            ),
            dependency_seed=repair.dependency_seed,
        )
        budget.remaining()
        alternate = state
        retain_recovery(durable=True)

    def probe_pinned_restore(target, advice):
        allowed = {
            path for path in paths if not _protected(path, task, new_test=False)
        }
        test_path = target[1].partition("::")[0].removeprefix("/work/").removeprefix("./")
        if (task.test_update or "").partition("::")[0] == test_path or test_path in allowed:
            return None, ""
        seed = pinned_stock_restore_seed(
            advice, original, current, allowed, minimum_bytes=_LONG_LOCALIZED_DEFINITION_BYTES
        )
        if seed is None:
            return None, ""
        sources, advisory = seed.sources, seed.advisory
        dependency_seed = seed.has_dependency_edits
        restored_patch = patch_for(original, sources)
        if not restored_patch or len(restored_patch) > executor.limits.patch_bytes:
            return None, advisory
        probe_budget = _pinned_restore_budget(budget, dependency_seed=dependency_seed)
        runs, static_runs = {}, {}
        unused_import_checks = set()
        try:
            checked = review_evidence(
                executor, task, archive, probe_budget, restored_patch, commands,
                pytest_runs=runs, static_runs=static_runs,
                **({"unused_import_checks": unused_import_checks} if dependency_seed else {}),
            )
            operator_failures = {key for key in checked if key[0] < len(task.pull_checks)}
            static_failures = _localized_static_failures(
                checked.keys() - operator_failures, static_runs, sources, allowed
            )
            if (
                not checked
                or static_failures is None
                or not qualified_stock_gain(
                    target, checked, runs, operator_failures, static_failures,
                    dependency_seed=dependency_seed,
                )
            ):
                return None, advisory
            selection = identify_stock_collection(target, patch, probe_budget, failed=True)
            if not selection or (
                identify_stock_collection(target, restored_patch, probe_budget) != selection
            ):
                return None, advisory
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
            return None, advisory
        return _RegressionRepair(
            target,
            sources,
            checked,
            set(),
            operator_failures,
            stock_origin=True,
            static_failures=static_failures,
            stock_collection_id=selection,
            dependency_seed=dependency_seed,
            unused_import_checks=frozenset(unused_import_checks),
        ), ""

    def verify_stock_candidate(target, sources, probe_budget, *, accept_partial=False):
        proposal = patch_for(original, sources)
        if not proposal or len(proposal) > executor.limits.patch_bytes:
            return None
        runs = {}
        checked = review_evidence(
            executor, task, archive, probe_budget, proposal, commands, pytest_runs=runs
        )
        if (
            checked.keys() < failures.keys()
            and _preserved_stock_gain(target, pytest_runs, runs, checked)
            and _verified_operator_runs(
                pytest_runs, runs, set(), task.pull_checks, operator_cases
            )
        ):
            return (sources, proposal, checked, runs), ""
        if accept_partial and is_isolated_stock_failure(target, checked, runs):
            note = tracked_assertion_note(target, sources, checked)
            if note:
                return (sources, proposal, checked, runs), note
        return None

    def verify_stock_gain(target, sources, probe_budget):
        verified = verify_stock_candidate(target, sources, probe_budget)
        return verified[0] if verified else None

    def focused_guard_trial(target, proposal, probe, *, selector=None):
        focused = focused_pytest((commands[target[0]],), (selector or target[1],))
        if focused is None:
            return False, ""
        focus = _GuardBudget(probe, wall_seconds=_STOCK_HUNK_SECONDS, reserve=0)
        runs = {}
        checked = review_evidence(
            executor, task, archive, focus, proposal, (focused,), pytest_runs=runs
        )
        run = runs.get(0)
        if run is None or run.deselected:
            return False, ""
        if not checked and run.passed == 1:
            return True, ""
        node = target[1]
        if not run.passed and run.failed == {node} and checked.keys() == {(0, node)}:
            return False, checked[0, node]
        return False, ""

    def round_limit_trial(target):
        pytest_checks = [
            index
            for index, command in enumerate(commands)
            if check_label(command, ()) == "pytest"
        ]
        if (
            not _named_operator_cases({target}, commands)
            or not _observed_operator_failures(
                pytest_runs, {target}, task.pull_checks, operator_cases
            )
            or any(pytest_runs.get(index) is None for index in pytest_checks)
            or budget.commands_remaining()
            < 2 * len(commands) + 2 * len(pytest_checks) + 3
            or budget.remaining() < 180
        ):
            return None, ""
        selected = _selected_operator_fixture(
            target[1], commands[target[0]], operator_cases, fixtures
        )
        if selected is None:
            return None, ""
        selector, fixture = selected
        owner = iteration_argument_owner(fixture, original, current)
        if owner is None:
            return None, ""
        allowed = {path for path in paths if not _protected(path, task, new_test=False)}
        guard = prepare_round_limit_guard(
            original, current, allowed, owner=owner, operator_fixture=fixture
        )
        if (
            guard is None
            or guard.sources.keys() != current.keys()
            or any(
                guard.sources[path] != current[path]
                for path in current.keys() - allowed
            )
        ):
            return None, ""
        proposal = patch_for(original, guard.sources)
        if not 0 < len(proposal) <= executor.limits.patch_bytes or proposal == patch:
            return None, ""
        advisory = excerpt(guard.advisory, 500)
        probe = _GuardBudget(budget, wall_seconds=220, reserve=60)
        try:
            passed, _ = focused_guard_trial(target, proposal, probe, selector=selector)
            if not passed:
                return None, advisory
            runs = {}
            checked = review_evidence(
                executor, task, archive, probe, proposal, commands, pytest_runs=runs
            )
            if checked or any(
                not _same_pytest_selection(pytest_runs[index], runs.get(index))
                for index in pytest_checks
            ):
                return None, advisory
            for index in pytest_checks:
                failed = index == target[0]
                node = selector if failed else None
                total = pytest_runs[index].passed + len(pytest_runs[index].failed)
                baseline = stock_collection(
                    executor,
                    task,
                    archive,
                    probe,
                    patch,
                    commands[index],
                    node,
                    total,
                    failed=failed,
                )
                if not baseline or baseline != stock_collection(
                    executor,
                    task,
                    archive,
                    probe,
                    proposal,
                    commands[index],
                    node,
                    total,
                ):
                    return None, advisory
            return (guard.sources, proposal, checked, runs), ""
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
            return None, advisory

    def verify_guard_order(target, sources):
        proposal = patch_for(original, sources)
        if not 0 < len(proposal) <= executor.limits.patch_bytes:
            return None, ""
        probe = _GuardBudget(budget, reserve=60)
        try:
            passed, detail = focused_guard_trial(target, proposal, probe)
            return (verify_stock_gain(target, sources, probe) if passed else None), detail
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
        return None, ""

    def verify_mixed_stock_gain(target, sources, probe):
        proposal = patch_for(original, sources)
        if not 0 < len(proposal) <= executor.limits.patch_bytes:
            return None
        operator_failures = {key for key in failures if key[0] < len(task.pull_checks)}
        runs = {}
        checked = review_evidence(
            executor, task, archive, probe, proposal, commands, pytest_runs=runs
        )
        if (
            not checked.keys() <= operator_failures
            or not _preserved_stock_gain(target, pytest_runs, runs, checked)
            or not _observed_operator_failures(
                runs, checked.keys(), task.pull_checks, operator_cases
            )
            or any(
                not _same_pytest_selection(pytest_runs.get(index), runs.get(index))
                for index, command in enumerate(commands)
                if check_label(command, ()) == "pytest"
            )
        ):
            return None
        selection = identify_stock_collection(target, patch, probe, failed=True)
        if selection and identify_stock_collection(target, proposal, probe) == selection:
            return sources, proposal, checked, runs
        return None

    def guard_order_with_operator_failures(
        target, operator_failures, *, generate=swap_traced_filters
    ):
        number, node = target
        baseline = pytest_runs.get(number)
        if (
            baseline is None
            or baseline.failed != {node}
            or not _named_operator_cases({target, *operator_failures}, commands)
            or commands[: len(task.pull_checks)] != task.pull_checks
            or not _observed_operator_failures(
                pytest_runs, operator_failures, task.pull_checks, operator_cases
            )
        ):
            return None, None
        exact = node.partition("[")[0].removeprefix("/work/").removeprefix("./")
        test_path = exact.partition("::")[0]
        line = stock_integer_assertion_failure_line(archive, node, failures[target])
        if (
            line is None
            or test_path in paths
            or (task.test_update or "").partition("::")[0] == test_path
        ):
            return None, None
        fixture = stock_pytest_excerpt(archive, commands[number], node, failures[target])
        if not fixture:
            return None, None
        stock_fixtures[target, failures[target]] = fixture
        try:
            trace = mock_call_diagnostic(
                executor,
                task,
                archive,
                budget,
                commands[number],
                node,
                patch,
                failure_line=line,
                reviewed_sources=(original, current),
            )
            loop = counter_loop_reuse(trace)
            if loop is None or loop.node != exact or loop.assertion_line != line:
                return None, None
            allowed = {
                path for path in paths if not _protected(path, task, new_test=False)
            }
            sources = generate(current, allowed, loop)
            if sources is None:
                return None, None
            proposal = patch_for(original, sources)
            if not 0 < len(proposal) <= executor.limits.patch_bytes:
                return None, None
            probe = _GuardBudget(budget, wall_seconds=180, reserve=60)
            passed, detail = focused_guard_trial(target, proposal, probe)
            if not passed:
                return None, (trace, detail) if detail else None
            return verify_mixed_stock_gain(target, sources, probe), None
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
        return None, None

    def save_verified_progress(checkpoint_patch, note, editor):
        nonlocal retained_failures, retained_identity, durable_witness, tradeoff
        budget.remaining()
        malformed.clear()
        native_ready.clear()
        no_progress.clear()
        retained_failures = failures
        retained_identity = _recovery_digest(task, fixtures, checkpoint_patch)
        if tradeoff and tradeoff.identity != tradeoff_identity():
            tradeoff = None
        if durable_witness and durable_witness.identity != witness_identity():
            durable_witness = None
        retain_recovery()
        task.save_checkpoint(
            DraftCheckpoint(
                _CHECKPOINT,
                task.base_sha,
                checkpoint_patch,
                (note,) if note else (),
                _GUIDANCE,
                (editor.route, editor.configured_model),
            )
        )

    def precision_stock_gain(
        target, signal, *, counterfactual="", paired_source="", verify=None
    ):
        node = target[1]
        exact = node.partition("[")[0].removeprefix("/work/").removeprefix("./")
        test_path = exact.partition("::")[0]
        diagnostic = failures[target]
        fixture = stock_fixtures.get((target, diagnostic), "")
        if (
            not fixture
            or signal.node != exact
            or signal.assertion_line != stock_assertion_failure_line(node, diagnostic)
            or not _named_operator_cases({target}, commands)
            or test_path in paths
            or (task.test_update or "").partition("::")[0] == test_path
        ):
            return None
        for site in (signal.owner, signal.mutation):
            source = current.get(site.path, "")
            if (
                site.path not in paths
                or _protected(site.path, task, new_test=False)
                or not source
                or hashlib.sha256(source.encode()).hexdigest() != site.source_digest
            ):
                return None
        symbol = signal.owner.path, signal.owner.name
        scoped = prepare_loop_line_range_edit(
            current,
            symbol,
            signal.body_line,
            signal.variable,
            signal.owner.line,
        )
        if scoped is None:
            return None
        snapshot, visible_rows, allowed_rows = scoped
        mutation = signal.mutation
        rows = current[mutation.path].splitlines()
        nearby = "\n".join(rows[max(0, mutation.line - 3) : mutation.line + 2])
        if wire_size(fixture) > 1700:
            window = stock_pytest_excerpt(
                archive,
                commands[target[0]],
                node,
                diagnostic,
                failure_window_only=True,
            )
            fixture = window + "\n" + excerpt(fixture, 1050)
        detail = (
            f"Selected failure: {node}\n"
            "Repair repeated logical processing with a small scoped edit. Identify the durable domain "
            "identity and when the record should count. Distinct records must still be processed. "
            "Preserve traversal and unrelated behavior. Do not use object IDs, clamp the numeric result, "
            "hardcode keys or test values, or change tests. Equal objects and logical duplicates are distinct concepts.\n"
            f"A paired replay observed the same object bound to `{signal.variable}` across "
            f"{signal.entries} loop entries at this site and {signal.mutations} mutations. "
            f"The pinned evaluation observed {signal.pinned_mutations} mutations in total. "
            "Object identity is diagnostic evidence within the draft evaluation.\n"
            f"Mutation source (quoted, read-only unless shown in editable rows): "
            f"{mutation.path}:{mutation.line} {mutation.name}\n{excerpt(nearby, 390)}\n"
            f"Immutable failing pytest input and setup (quoted):\n{fixture}\n"
            f"Observed pytest failure (quoted):\n{excerpt(diagnostic, 450)}\n"
        )
        if counterfactual:
            detail += (
                f"Verified order-only counterfactual (quoted observation):\n{counterfactual}\n"
                "Compare stable record keys and the current and pinned processing. Consider when a "
                "logical record should create or increment the metric. Keep unrelated rules intact.\n"
            )
        context = (
            f"Pinned caller excerpt (quoted, read-only):\n{excerpt(paired_source, 410)}\n"
            if counterfactual and paired_source
            else ""
        )
        ending = (
            line_range_prompt(snapshot, visible_rows=visible_rows) + LINE_RANGE_ENDING
        )
        messages = [
            {"role": "system", "content": _LINE_RANGE_SYSTEM},
            {"role": "user", "content": detail + context + ending},
        ]
        if context and wire_size(messages) > 7500:
            messages[-1]["content"] = detail + ending
        if wire_size(messages) > 7500:
            return None
        try:
            completion = budget.complete(
                executor.gateway,
                "coding",
                messages,
                after_provider=actor.route,
                max_output_tokens=2048,
                retry_oversize=False,
                timeout_cap=_LOOP_PRECISION_MODEL_SECONDS,
            )
            editor = RouteEvidence.from_completion(completion)
            if editor.role != "coding" or not editor.configured_model:
                return None
            proposed = apply_line_range_edit(
                completion.content,
                current,
                snapshot,
                allowed_rows=allowed_rows,
            )
            probe = _GuardBudget(budget, reserve=_LOOP_PRECISION_RESERVE_SECONDS)
            if verify is not None:
                qualification = verify(target, proposed, probe)
                verified = (qualification, "") if qualification is not None else None
            else:
                verified = verify_stock_candidate(
                    target,
                    proposed,
                    probe,
                    accept_partial=not counterfactual,
                )
            if verified is not None:
                return *verified, editor
        except Cancelled:
            raise
        except (ExecutionError, ProviderExhausted, ValueError):
            budget.remaining()
        return None

    def try_semantic_record_key(target, trace, focused_failure, *, mixed=False):
        nonlocal current, patch, failures, pytest_runs
        signal = counter_loop_reuse(trace)
        if (
            signal is None
            or not focused_failure
            or budget.remaining() < 200
            or budget.commands_remaining() < 3 * len(commands) + 6
            or not stock_missing_count_after_reorder(
                archive, target[1], failures[target], focused_failure, signal.assertion_line
            )
        ):
            return False
        allowed = {path for path in paths if not _protected(path, task, new_test=False)}
        sources = guard_traced_counter_once(current, signal, allowed)
        if sources is None:
            return False
        _LOG.info("Checking the existing record key against repeated diagnostic events")
        proposal = patch_for(original, sources)
        if not 0 < len(proposal) <= executor.limits.patch_bytes:
            return False
        probe = _GuardBudget(budget, wall_seconds=180, reserve=60)
        try:
            passed, _ = focused_guard_trial(target, proposal, probe)
            if not passed:
                return False
            verify = verify_mixed_stock_gain if mixed else verify_stock_gain
            verified = verify(target, sources, probe)
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
            return False
        if verified is None:
            return False
        current, patch, failures, pytest_runs = verified
        start_new_draft(target)
        _LOG.info("Record-key trial passed the repository checks with %d review case(s) remaining", len(failures))
        return True

    def try_guard_precision(target, trace, focused_failure, *, verify=None):
        nonlocal current, failures, patch, pytest_runs, actor, editor
        nonlocal precision_attempted, submitted
        needs_regression = task.require_new_tests and not has_runnable_test(
            new_path, current[new_path].encode()
        )
        if (
            precision_attempted
            or not focused_failure
            or actor is None
            or actor.role != "coding"
            or not actor.configured_model
            or target not in failures
            or (failures.keys() != {target} and verify is None)
            or executor.verification_runner is executor.runner
            or min(max_proposals - submitted, budget.model_requests_remaining())
            < 1 + int(needs_regression)
            or budget.commands_remaining() < 3 * len(commands) + 4
            or budget.remaining()
            < _LOOP_PRECISION_MODEL_SECONDS
            + _LOOP_PRECISION_RESERVE_SECONDS
            + _GUARD_MIN_REMAINING
            or (signal := counter_loop_reuse(trace)) is None
        ):
            return False
        observation = stock_missing_count_after_reorder(
            archive, target[1], failures[target], focused_failure, signal.assertion_line
        )
        if not observation:
            return False
        site = counter_caller_site(trace)
        pinned = (
            site.excerpts.partition("\nDraft caller source")[0]
            if site is not None
            else ""
        )
        precision_attempted = True
        before = budget.model_requests_remaining()
        _LOG.info(
            "Trying scoped precision after a verified filter-order counterfactual"
        )
        proposed = precision_stock_gain(
            target,
            signal,
            counterfactual=observation,
            paired_source=pinned,
            verify=verify,
        )
        submitted += before - budget.model_requests_remaining()
        if proposed is None:
            return False
        (_, _, checked, runs), _, _ = proposed
        if not checked.keys() < failures.keys() or not _preserved_stock_gain(
            target, pytest_runs, runs, checked
        ):
            return False
        (current, patch, failures, pytest_runs), note, actor = proposed
        editor = actor
        start_new_draft(target)
        if failures or needs_regression:
            save_verified_progress(patch, note, editor)
        return True

    def interactive_stock_gain(target, symbol, probe):
        reviewed = {
            path: current[path]
            for path in paths
            if not _protected(path, task, new_test=False)
        }
        if not reviewed:
            return None
        if task.require_new_tests and not _protected(new_path, task, new_test=True):
            reviewed[new_path] = current[new_path]
        draft = DraftCheckpoint(
            _CHECKPOINT, task.base_sha, patch, guidance=_GUIDANCE,
            last_editor=(actor.route, actor.configured_model) if actor and actor.configured_model else None,
        )
        command = commands[target[0]]
        diagnostic = failures[target]
        stock_fixture = stock_pytest_excerpt(archive, command, target[1], "")
        window = (
            stock_pytest_excerpt(
                archive, command, target[1], diagnostic, failure_window_only=True,
            )
            if wire_size(stock_fixture) > 1400 else ""
        )
        inferred_symbol, example = interactive_stock_focus(
            original, current, stock_fixture + window, fixtures
        )
        symbol = symbol or inferred_symbol
        focused = focused_pytest((command,), (target[1],))
        command_label, argv = ("Focused", focused) if focused else ("Configured", command)
        instructions = (
            "The saved draft passes the operator review checks. Use the workspace tools to "
            "reproduce the tracked test, try a focused fix, and preserve all passing behavior. "
            "Use run to read the complete files when needed."
            f"\nTracked pytest node: {target[1]}"
            f"\n{command_label} pytest argv: {json.dumps(argv)}"
            f"\nEdit only these paths: {json.dumps(sorted(reviewed))}"
        )
        if example:
            instructions += "\nPassing operator suite example (quoted):\n" + example
        stock_heading = "\nTracked test and fixture setup (quoted):\n"
        sections = [
            ("\nLast failure (quoted):\n", excerpt(diagnostic, 650)),
            ("\nExecution observations (quoted):\n", compact_observations(mock_diagnostics.get(target, ""), 1080)),
        ]
        if symbol is not None:
            sections.append(("\nLocalized method in the current draft (quoted):\n", "::".join(symbol)))
        sections.append((stock_heading, stock_fixture))
        if symbol is not None and (definition := prepare_line_range_edit(current, symbol)):
            sections.append(("\nCurrent method source (quoted):\n", "\n".join(definition.lines)))
        sections.append(("\nOriginal review (quoted):\n", excerpt(task.instructions, 650)))
        # Leave room for checkpoint and operator-command annotations and complete
        # tool results even on the smallest supported local model prompt.
        for heading, section in sections:
            if not section:
                continue
            prompt = {"role": "user", "content": instructions + heading}
            room = _INTERACTIVE_INSTRUCTIONS_BYTES - wire_size([prompt])
            if room < 160:
                break
            if heading == stock_heading and window and wire_size(section) - 2 > room:
                section = window + "Use run to read the complete tracked test and fixture setup.\n"
            bounded = excerpt(section, room)
            while wire_size([{**prompt, "content": instructions + heading + bounded}]) > _INTERACTIVE_INSTRUCTIONS_BYTES:
                room //= 2
                bounded = excerpt(section, room)
            instructions += heading + bounded
        initial_tools = probe.tools_left
        scoped_actions = Counter()
        interactive_task = replace(
            task,
            instructions=instructions,
            checkpoints={_CHECKPOINT: draft},
            save_checkpoint=None,
            save_solver_recovery=None,
            review_recovery=None,
            repair_failure=(),
            context=None,
            test_update=None,
        )
        try:
            _, _, editor = executor._rollout(
                interactive_task,
                archive,
                probe,
                SYSTEM_PROMPT,
                "coding",
                clarification=False,
                checkpoint_key=_CHECKPOINT,
                checkpoint_guidance=_GUIDANCE,
                scoped_sources=reviewed,
                scoped_action_counts=scoped_actions,
            )
            sources = {**current, **reviewed}
            # Prove the production fix with the accepted test selection first.
            production = (
                {**sources, new_path: original[new_path]}
                if new_path in sources
                else sources
            )
            if (
                not any(production[path] != current[path] for path in paths)
                or editor is None
                or not editor.configured_model
            ):
                _LOG.info(
                    "Interactive review ended without a new attributable production edit"
                )
                return None
            verified = verify_stock_candidate(
                target, production, budget, accept_partial=True
            )
            if verified is not None:
                checked, note = verified
                return checked, editor, note, sources.get(new_path)
            _LOG.info(
                "The interactive edit still had outstanding verification failures"
            )
        except Cancelled:
            raise
        except (ExecutionError, ProviderExhausted, ValueError, OSError) as error:
            _LOG.info("Interactive review ended with %s", type(error).__name__)
            budget.remaining()
        finally:
            _LOG.info(
                "Interactive review used %d model turns and %d tool actions "
                "(exact edits succeeded=%d rejected=%d, shell actions=%d)",
                probe.limits.rollout_requests - probe.turns_left,
                initial_tools - probe.tools_left,
                scoped_actions["successful_edits"],
                scoped_actions["rejected_edits"],
                scoped_actions["shell_actions"],
            )
        return None

    def reconcile_stock_gain(
        target, proposed_sources, regressions=None, *, static_locations=()
    ):
        if (
            budget.remaining() < _GUARD_MIN_REMAINING
            or budget.commands_remaining() < 3 * len(commands) + 1
            or proposed_sources.keys() != current.keys()
        ):
            return None
        changed = [path for path in current if current[path] != proposed_sources[path]]
        allowed = {path for path in paths if not _protected(path, task, new_test=False)}
        if not changed or any(path not in allowed for path in changed):
            return None
        try:
            candidates = ()
            if len(changed) == 1 and not static_locations:
                path = changed[0]
                candidates = tuple(
                    {**current, path: source}
                    for source in guarded_sources(current[path], proposed_sources[path])
                )
            selective = not candidates and regressions is not None
            if selective:
                locations = {
                    (path.removeprefix("/work/").removeprefix("./"), row)
                    for (number, _), diagnostic in regressions.items()
                    for path, row in check_locations(commands[number], diagnostic)
                }
                candidates = selective_reverts(
                    current,
                    proposed_sources,
                    allowed,
                    locations,
                    static_locations=static_locations,
                )
            probe_budget = _GuardBudget(budget)
            for sources in candidates:
                verified = verify_stock_gain(target, sources, probe_budget)
                if verified is not None and (not static_locations or not verified[2]):
                    if selective:
                        _LOG.info(
                            "Selective reversion preserved the verified stock gain and review cases"
                        )
                    return verified
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
        return None

    def compose_stock_gain(target, proposal):
        nonlocal stock_edits, composition_attempted
        if (
            composition_attempted
            or budget.remaining() < _COMPOSITION_MIN_REMAINING
            or budget.commands_remaining() < 2 * len(commands) + 1
        ):
            return None
        allowed = {path for path in paths if not _protected(path, task, new_test=False)}
        edits = small_line_edits(current, proposal, allowed)
        if not edits:
            return None
        previous, stock_edits = stock_edits, edits
        composed = compose_line_edits(current, previous, edits)
        if composed is None:
            return None
        composition_attempted, stock_edits = True, ()
        try:
            return verify_stock_gain(target, composed, _GuardBudget(budget))
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
            return None

    def isolate_stock_hunk(target, proposal, checked, runs, introduced, static_runs):
        nonlocal hunk_isolation_attempted
        number, node = target
        before, after = pytest_runs.get(number), runs.get(number)
        if (
            hunk_isolation_attempted
            or not _named_operator_cases({target}, commands)
            or not _same_pytest_selection(before, after)
            or before.failed != {node}
            or after.failed != {node}
            or not _verified_operator_runs(
                pytest_runs, runs, introduced, task.pull_checks, operator_cases
            )
            or budget.remaining() < _COMPOSITION_MIN_REMAINING
            or budget.commands_remaining() < 2 * len(commands) + _STOCK_HUNK_PROBES + 1
        ):
            return None
        allowed = {path for path in paths if not _protected(path, task, new_test=False)}
        if not introduced and not _localized_static_failures(
            checked.keys() - {target}, static_runs, proposal, allowed
        ):
            return None
        focused = focused_pytest((commands[number],), (node,))
        if focused is None:
            return None
        try:
            probe_budget = _GuardBudget(budget)
            for sources in individual_line_edits(
                current, proposal, allowed, limit=_STOCK_HUNK_PROBES
            ):
                trial = patch_for(original, sources)
                if not trial or len(trial) > executor.limits.patch_bytes:
                    continue
                hunk_isolation_attempted = True
                focus_budget = _GuardBudget(
                    probe_budget, wall_seconds=_STOCK_HUNK_SECONDS, reserve=0
                )
                focused_runs = {}
                errors = review_evidence(
                    executor,
                    task,
                    archive,
                    focus_budget,
                    trial,
                    (focused,),
                    pytest_runs=focused_runs,
                )
                run = focused_runs.get(0)
                if run is None or run.deselected:
                    continue
                if (run.passed == 1 and not errors) or (
                    not run.passed
                    and run.failed == {node}
                    and errors.keys() == {(0, node)}
                    and tracked_assertion_note(target, sources, {target: errors[0, node]})
                ):
                    return verify_stock_candidate(
                        target, sources, probe_budget, accept_partial=True
                    )
        except Cancelled:
            raise
        except (ExecutionError, ValueError):
            budget.remaining()
        return None

    planning_attempts = {}
    planned_private_methods = set()
    retry_advice = {}
    line_retries = {}
    line_routes = {}
    iteration_attempted = set()
    anchored_line_attempted = set()
    raw_nested_attempts = {}

    def retry_numbered_route(target, line_retry, editor, *, malformed=False, after=None, range_error=None, checked_candidate=None):
        if (
            line_retry is None
            or line_retry.checkout is not current
            or target not in failures
            or (len(line_retry.excluded_models) >= 3 and not (line_retry.direct_caller_only and checked_candidate is not None))
            or not editor.configured_model
        ):
            return
        model = editor.route, editor.configured_model
        if model in line_retry.excluded_models:
            return
        excluded = (*line_retry.excluded_models, model)
        if line_retry.direct_caller_only and checked_candidate is not None and after is not None:
            followup = _checked_caller_retry(line_retry, checked_candidate, target, failures, after, excluded[-3:])
            if followup:
                line_retries[target] = followup
            return
        if line_retry.checkpoint_context and len(line_retry.alternatives) == 1:
            caller, = line_retry.alternatives
            visible = frozenset(range(1, len(caller.lines) + 1))
            allowed = frozenset(range(caller.first_body_line, len(caller.lines) + 1))
            if malformed:
                result = "The previous proposal could not be applied to the selected definition."
                if isinstance(range_error, ScopedRangeRejection):
                    result += " " + str(range_error)
            else:
                result = "The previous proposal did not improve verification."
                if after is not None:
                    added = len(after.keys() - failures.keys())
                    if added:
                        result += f" It introduced {added} additional failing check(s)."
                    observed = after.get(target, "").partition("\nObserved pytest failure: ")[2]
                    if observed:
                        result += "\nObserved pytest failure on the previous proposal: " + excerpt(observed, 350)
            context = line_retry.checkpoint_context + result + "\nWork only on the direct caller shown below.\n"
            text = context + line_range_prompt(caller, visible_rows=visible, full_method=True) + _SINGLE_METHOD_ENDING
            native = context + line_range_prompt(caller, visible_rows=visible, native=True, full_method=True)
            if wire_size([{"role": "system", "content": _LINE_RANGE_SYSTEM}, {"role": "user", "content": text}]) <= RETRY_MESSAGE_BYTES:
                line_retries[target] = replace(
                    line_retry, snapshot=caller, allowed_rows=allowed, prompt=text, native_prompt=native,
                    alternatives=(), checkpoint_context="", direct_caller_only=True, excluded_models=excluded,
                    caller_context=context,
                    native_ready=False, native_attempted=False,
                )
                return
        if line_retry.checkpoint_method_edit and isinstance(range_error, ScopedRangeRejection):
            note = f"Previous edit rejected: {range_error}\n"
            if note not in line_retry.prompt:
                prompt = line_retry.prompt.removesuffix(LINE_RANGE_ENDING) + note + LINE_RANGE_ENDING
                if wire_size([{"role": "system", "content": _LINE_RANGE_SYSTEM}, {"role": "user", "content": prompt}]) <= RETRY_MESSAGE_BYTES:
                    line_retry = replace(line_retry, prompt=prompt, native_prompt=line_retry.native_prompt + note)
        line_retries[target] = replace(
            line_retry, excluded_models=excluded, native_ready=line_retry.native_ready or malformed,
        )

    def queue_review_anchored_retry(target):
        number, node = target
        run = pytest_runs.get(number)
        whole_method = (
            current is resumed_operator_checkout and failures.keys() == {target}
            and run is not None and run.failed == {node} and not run.deselected
            and any(control and control.passed and not control.deselected for control in pytest_runs.values())
            and 0 <= number < len(task.pull_checks)
            and not _trusted_operator_node(node, commands[number], operator_cases)
        )
        if (
            not production_regression
            or (not whole_method and malformed.get(target, 0) < RECOVERY_MODE_THRESHOLD)
            or not 0 <= number < len(task.pull_checks)
            or target not in failures
            or executor.verification_runner is executor.runner
            or run is None or node not in run.failed
            or commands[number] != task.pull_checks[number]
            or not _named_operator_cases({target}, commands)
            or _protected(task.regression_edit_path, task, new_test=False)
        ):
            return
        revision = target, hashlib.sha256(patch).digest()
        if revision in anchored_line_attempted:
            return
        anchored_line_attempted.add(revision)
        retry = _review_anchored_line_retry(
            _draft_review(review, original, current), current, task.regression_edit_path,
            node, failures[target], whole_method=whole_method,
        )
        if retry is not None:
            continuations.pop(target, None)
            line_retries[target] = retry

    def queue_iteration_retry(target, snapshot):
        number, node = target
        run = pytest_runs.get(number)
        if (
            snapshot is not current
            or target in iteration_attempted
            or not 0 <= number < len(task.pull_checks)
            or target not in failures
            or executor.verification_runner is executor.runner
            or run is None
            or node not in run.failed
            or commands[number] != task.pull_checks[number]
            or not _named_operator_cases({target}, commands)
            or not _trusted_operator_node(node, commands[number], operator_cases)
        ):
            return False
        fixture = _fixture_excerpt(fixtures, node, selected_only=True)
        windows = iteration_edit_windows(fixture, original, current)
        if windows is None:
            return False
        path, _ = windows.symbol
        if path not in paths or _protected(path, task, new_test=False):
            return False
        scope = prepare_window_line_range_edit(current, windows.symbol, windows.spans)
        if scope is None:
            return False
        exact, visible, allowed = scope
        prompt = (
            f"Selected failure: {node}\n{windows.note}\n"
            "Repair the public iteration limit in the current traversal. Preserve the other "
            "passing behavior and edit only the shown body rows.\n"
            f"Selected operator fixture (quoted):\n{excerpt(fixture, 1200)}\n"
            f"Verified failure (quoted):\n{excerpt(failures[target], 600)}\n"
            + (
                f"Related definition (read-only):\n{windows.related}\n"
                if windows.related else ""
            )
            + line_range_prompt(exact, visible_rows=visible)
            + LINE_RANGE_ENDING
        )
        if wire_size([{"role": "user", "content": prompt}]) > 7500:
            return False
        iteration_attempted.add(target)
        continuations.pop(target, None)
        line_retries[target] = _LineRetry(None, exact, allowed, prompt)
        return True
    stock_fixtures = {}
    mock_diagnostics = {}
    counter_sites = {}
    max_proposals = min(36, budget.model_requests_remaining())
    editor = actor
    submitted = 0
    pending = {}
    interactive_target = interactive_symbol = None
    interactive_misses = Counter()
    unlocalized_plans = set()
    interactive_attempted = False
    stock_hedge = None
    hedge_used = False
    hedge_pair = set()
    retired = None
    continuations = {}
    stock_edits = ()
    composition_attempted = False
    hunk_isolation_attempted = False
    regression_repair = None
    if tradeoff and budget.commands_remaining() >= len(commands) * 2 + 8 and budget.remaining() >= 90:
        regression_repair = restore_tradeoff()
        if not regression_repair:
            tradeoff = None
            retain_recovery(durable=True)
    if regression_repair is None:
        regression_repair = restore_private_progress()
    repair_attempted = {regression_repair.origin} if regression_repair and regression_repair.private_symbol else set()
    pinned_restore_attempted = regression_repair is not None
    pinned_restore_advice = {}
    eager_source = current
    eager_target = next(iter(failures)) if len(failures) == 1 else None
    if not (
        eager_target is not None
        and draft is not None
        and actor is not None
        and actor.role == "coding"
        and actor.configured_model
        and retained_recovery is not None
        and retained_recovery.identity == retained_identity
        and max(malformed.get(eager_target, 0), no_progress.get(eager_target, 0))
        >= RECOVERY_MODE_THRESHOLD
        and eager_target[0] >= len(task.pull_checks)
        and _named_operator_cases({eager_target}, commands)
        and (eager_run := pytest_runs.get(eager_target[0])) is not None
        and eager_run.failed == {eager_target[1]}
        and _verified_operator_runs(
            pytest_runs, pytest_runs, set(), task.pull_checks, operator_cases
        )
    ):
        eager_target = None
    eager_trace, eager_advisory = None, ""
    precision_attempted = False
    mixed_guard_attempted = False
    round_limit_snapshot = None
    round_limit_advice = {}
    unavailable = {}
    test_draft = test_seed.source if test_seed else current.get(new_path, "")
    test_diagnostic = (
        "This independently verified partial draft is authoring context. Preserve its "
        "passing cases and submit a complete module restoring these omitted coverage cases: "
        + ", ".join(test_seed.omitted)
        if test_seed else ""
    )
    test_rejection = None
    test_ancestor_feedback = ""
    test_passed_ancestor = False
    fixture_requests = (
        explicit_fixture_requests(review.feedback)
        if production_regression or task.inline_review_threads >= 2 else ()
    )
    known_fixture_omissions = {}
    test_case_runs = {}
    focused_generated = focused_pytest(task.checks, (new_path,), verbose=True) if new_path else None
    test_checkout_snapshot = None
    test_checkout_digest = None
    if test_seed:
        test_case_runs["focused"] = _PytestRun(
            len(test_seed.passing), frozenset(), 0,
            frozenset(f"{new_path}::{name}" for name in test_seed.passing),
        )
        test_checkout_snapshot = current
        test_checkout_digest = hashlib.sha256(patch).digest()
    test_format_revision = None
    test_format_failures = 0
    test_setup_failures = 0
    prior_test_attempted = False
    prior_test_source = ""
    prior_test_fallback = None
    helper_setup = None
    helper_attempts = 0
    helper_outcomes = Counter()
    helper_setup_misses = 0
    helper_feedback = ""
    helper_function = ""
    helper_bad_model = ()
    test_model_failures = Counter()
    test_tool_attempts = set()
    test_line_attempts = set()
    test_raw_attempts = {}
    repository_example = None
    ancestor_delta = ""
    repository_checkout = None
    extended_context = ""
    repository_usage = ""
    repository_models = ""
    attempted_seed_cases = set()
    ancestor_assertion_hints = {}

    test_first_attempted = False
    test_first_witness = None
    test_first_format_misses = 0
    test_first_runtime = None

    def author_red_regression(routes, after_provider, runnable_drafts):
        nonlocal test_first_format_misses, test_first_runtime
        # Only repository context and approved setup helpers are offered. The
        # independent operator scenario and its diagnostic are kept out.
        public = {path: current[path] for path in paths}
        request = " ".join(body for _, _, body in review.feedback)
        example, usage = repository_pytest_context(
            task.regression_archive, public, request, exclude_generated=True,
        )
        query = words(request)
        api = "\n".join(
            _regression_api(path, content, query, 1500, anchors=[
                line for location, line, _ in review.feedback if location == path
            ])[0] for path, content in public.items()
        )
        setup = prior_generated_fixture_context(
            archive, task.regression_archive, new_path, public, require_identical=True,
        )
        examples = "\n".join(filter(None, (setup, example, usage)))
        protocol = production_protocol_example(
            setup.partition("\n")[2], public, query,
            review_anchors=tuple((path, line) for path, line, _ in review.feedback),
        )

        prompt = (
            f"Test-first regression for {new_path}. Write exactly one original offline pytest test "
            "that demonstrates the requested public behavior. Both current and historical production "
            "are expected to fail its ordinary assertion. Return the complete Python module including imports.\n"
            + excerpt(request, 1200) + "\nProduction APIs:\n" + api
            + "\nRepository setup examples:\n" + excerpt(examples, 3200)
            + ("\nProduction call and response handling:\n" + protocol if protocol else "")
        )
        if runnable_drafts:
            previous, outcomes = next(reversed(runnable_drafts.items()))
            prompt += (
                "\nPrevious authored test (quoted):\n" + excerpt(previous, 4000)
                + "\nMeasured executions: " + ", ".join(outcomes)
                + ". Strengthen the scenario and assertions so the missing behavior is observable. "
                "Preserve relevant passing checks. Return the complete revised module."
            )
        messages = [{"role": "system", "content": _SYSTEM_PREFIX + _SYSTEM_SUFFIX},
                    {"role": "user", "content": prompt}]
        assembly = None
        if test_first_format_misses and fixtures and not runnable_drafts:
            assembly = operator_test_assembly(
                fixtures, archive, task.regression_archive,
                {path: public[path] for path in public if not _protected(path, task, new_test=False)},
            )
            if assembly:
                previous, feedback = test_first_runtime or ("", "")
                scoped = _operator_function_messages(assembly, review, public, new_path, "", test_first=True,
                                                     prior_function=previous, feedback=feedback)
                if wire_size(scoped) <= 12 * 1024:
                    messages = scoped
                    _LOG.info("Requesting a test-first function using approved setup helpers")
                else:
                    assembly = None
        remaining = budget.remaining()
        result = budget.complete(executor.gateway, "coding", messages, max_output_tokens=2048, retry_oversize=False,
                                 exclude=tuple(routes), after_provider=after_provider,
                                 timeout_cap=remaining - max(60, remaining / 3))
        model = result.route, result.configured_model
        if model in routes or not result.configured_model:
            _LOG.info("Test-first authoring rejected duplicate or unidentified route")
            return None
        routes.append(model)
        max_bytes = min(8192, executor.limits.patch_bytes)
        if assembly:
            try:
                source = assembly.assemble(result.content, max_bytes=max_bytes)
            except ValueError:
                source = None
        else:
            source = _complete_test_module(result.content, new_path, max_bytes)
        if not source or not _one_test_draft(source):
            test_first_format_misses += 1
            reply = result.content or ""
            _LOG.info("Test-first rejected %s format: characters=%d fences=%d test declarations=%d",
                      "function" if assembly else "module", len(reply), reply.count("```"),
                      len(re.findall(r"(?m)^def test_\w+\(", reply)))
            return None
        if source in runnable_drafts:
            _LOG.info("Test-first author repeated a previously measured green test")
            return None
        outcomes = []
        runtime = []
        witness = prove_red_regression(source, outcomes=outcomes, runtime=runtime if assembly else None)
        if assembly and runtime:
            test_first_runtime = (result.content, runtime[0] + "\nCorrect the setup or API usage and retain the requested behavioral assertions.")
        if not witness and len(outcomes) == 2 and any(item.endswith(" passed") for item in outcomes):
            runnable_drafts[source] = tuple(outcomes)
        return witness

    def prove_red_regression(source, *, outcomes=None, runtime=None):
        if _complete_test_module(source, new_path, min(8192, executor.limits.patch_bytes)) != source or not _one_test_draft(source):
            return None
        proof = _GuardBudget(budget, wall_seconds=45)

        def exact_bytes(workspace, _tree):
            staged = executor._command(workspace, proof, ("cat", "--", new_path), cap=executor.limits.patch_bytes)
            if staged.stdout != source.encode():
                raise ValueError("Staged regression differs from authored bytes")

        assertions = []
        for stage, revision, changes in (
            ("checkpoint", archive, patch_for(original, {**current, new_path: source})),
            ("pinned ancestor", task.regression_archive, patch_for({new_path: ""}, {new_path: source})),
        ):
            runs = {}
            failed = review_evidence(executor, task, revision, proof, changes, (focused_generated,),
                                     pytest_runs=runs, on_staged=exact_bytes)
            run = runs.get(0)
            if not run or len(run.failed) != 1:
                outcome = "green" if run and run.passed == 1 and not failed and not run.deselected else "incomplete or setup error"
                _LOG.info("Test-first %s verification: %s", stage, outcome)
                if outcome == "green" and outcomes is not None:
                    outcomes.append(stage + " passed")
                    continue
                return None
            observed = next(_focused_test_assertions(new_path, source, run, failed), None)
            if observed is None:
                _LOG.info("Test-first %s verification: failure outside test assertion", stage)
                if runtime is not None and (hint := _focused_runtime_hint(new_path, source, run, failed)):
                    runtime.append(stage + " execution: " + hint)
                    _LOG.info("Test-first function reached a correctable runtime failure in its authored body")
                return None
            _LOG.info("Test-first %s verification: red at a verified test assertion", stage)
            assertions.append(observed)
            if outcomes is not None:
                outcomes.append(f"{stage} failed by assertion at line {observed[1]}")
        if len(assertions) != 2 or assertions[0][0] != assertions[1][0]:
            _LOG.info("Test-first verification rejected mismatched collected cases")
            return None
        rows = source.splitlines()
        diagnostics = [
            f"{revision} assertion at line {row}: {excerpt(rows[row - 1].strip(), 250)}\n"
            + excerpt(rewritten, 350)
            for revision, (_, row, rewritten) in zip(("Checkpoint", "Pinned ancestor"), assertions)
        ]
        return source, ("An independently authored repository test failed on both revisions.\n"
                        + "\n".join(diagnostics)
                        + "\nOriginal test setup and scenario (quoted):\n" + excerpt(source, 1600))

    def failed_assertion_hint(source, run, fixed_failures):
        if (
            not run or not run.assertion_reached or len(run.failed) != 1
            or not focused_generated
        ):
            return ""
        fixed = next(_focused_test_assertions(new_path, source, run, fixed_failures), None)
        if fixed is None:
            return ""
        node = fixed[0]
        name = node.split("[", 1)[0].rsplit("::", 1)[-1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name):
            return ""
        projected = patch_for({new_path: ""}, {new_path: source})
        revision = hashlib.sha256(patch + b"\x00" + projected).digest()
        if revision in ancestor_assertion_hints:
            return ancestor_assertion_hints[revision]
        if (
            len(ancestor_assertion_hints) >= 2 or not budget.model_requests_remaining()
            or budget.commands_remaining() < len(commands) + 4 or budget.remaining() < 45
            or getattr(executor.verification_runner, "network", "none") != "none"
        ):
            return ""
        ancestor_assertion_hints[revision] = ""
        probe = _GuardBudget(budget, wall_seconds=20)

        def check_test_bytes(workspace, _tree):
            result = executor._command(
                workspace, probe, ("cat", "--", new_path), cap=executor.limits.patch_bytes,
            )
            if result.stdout != source.encode():
                raise ValueError("Historical replay did not stage the original test bytes")

        old_runs = {}
        try:
            old_failures = review_evidence(
                executor, task, task.regression_archive, probe, projected,
                (focused_generated,), pytest_runs=old_runs, on_staged=check_test_bytes,
            )
        except Cancelled:
            raise
        except (ExecutionError, OSError, ValueError):
            return ""
        older = old_runs.get(0)
        if older is None or older.deselected or older.named_cases != run.named_cases:
            return ""
        if not old_failures and older.passed == 1:
            hint = (
                f"Pinned comparison for {name} failed on the fixed revision and passed on the pinned ancestor. "
                "Recheck the assertion against the reviewed behavior."
            )
        elif old := next(_focused_test_assertions(new_path, source, older, old_failures), None):
            hint = _assertion_comparison_hint(name, fixed, old)
        else:
            return ""
        ancestor_assertion_hints[revision] = hint
        return hint

    def verified_prior_seed():
        if budget.commands_remaining() < len(commands) + 6 or budget.remaining() < 35:
            return None
        previous = prior_generated_test_seed(
            archive, task.regression_archive, new_path,
            {path: current[path] for path in paths if not _protected(path, task, new_test=False)},
            " ".join(body for _, _, body in review.feedback),
        )
        if previous is None:
            return None
        path, name, source = previous
        projected = patch_for({new_path: ""}, {new_path: source})
        proof = _GuardBudget(budget, wall_seconds=25)
        try:
            for archive_revision, changes in (
                (task.regression_archive, projected),
                (archive, patch_for(original, {**current, new_path: source})),
            ):
                runs = {}
                if review_evidence(
                    executor, task, archive_revision, proof, changes,
                    (focused_generated,), pytest_runs=runs,
                ) or not (
                    (run := runs.get(0)) and run.passed == 1
                    and _covers_generated_seed(run, new_path, (name,))
                ):
                    return None
        except Cancelled:
            raise
        except (ExecutionError, LimitExceeded):
            budget.remaining()
            return None
        return path, name, source, run

    def count_test_model_failure(snapshot, editor, categories):
        if snapshot is current and test_model_failures is not None and editor.configured_model:
            pair = editor.route, editor.configured_model
            kind = tuple(sorted(categories)) or ("verification",)
            key = pair, kind
            test_model_failures[key] = min(test_model_failures[key] + 1, _MODEL_FAILURE_THRESHOLD)

    def lone_both_pass_case():
        if not (
            production_regression and len(review.feedback) == 1 and not review.test_only
            and test_checkout_snapshot is current and test_seed is None and test_passed_ancestor
            and original[new_path] == current[new_path] == ""
            and not _protected(new_path, task, new_test=True) and _one_test_draft(test_draft)
        ):
            return None
        run = test_case_runs.get("focused")
        if run and run.passed == 1 and not run.failed and not run.deselected and run.named_cases:
            if len(run.named_cases) == 1:
                node = next(iter(run.named_cases))
                if node.startswith(new_path + "::"):
                    return node
        return None

    def retain_test_seed(candidate, runs):
        nonlocal test_seed
        if (
            task.save_solver_recovery is None
            or executor.runner is executor.verification_runner
            or _recovery_digest(task, fixtures, patch) != retained_identity
            or any(candidate[path] != current[path] for path in paths)
        ):
            return
        try:
            seed = _project_generated_seed(new_path, candidate[new_path], runs.get("focused"))
        except (ValueError, SyntaxError, RecursionError):
            return
        if seed is None or seed.passing in attempted_seed_cases:
            return
        all_names = set(seed.passing + seed.omitted)
        if test_seed and not (
            set(test_seed.passing) < set(seed.passing)
            and set(test_seed.passing + test_seed.omitted) <= all_names
        ):
            return
        attempted_seed_cases.add(seed.passing)
        projection = patch_for(original, {**current, new_path: seed.source})
        try:
            verified = executor._verify(task, archive, budget, projection)
        except Cancelled:
            raise
        except ExecutionError:
            budget.remaining()
            return
        run = _focused_generated_run(verified.checks, focused_generated)
        if not _covers_generated_seed(run, new_path, seed.passing) or run.passed != len(seed.passing):
            return
        test_seed = seed
        retain_recovery(durable=True)
        _LOG.info("Saved independently verified authoring context for %d generated test cases", run.passed)

    def start_new_draft(target):
        nonlocal interactive_target, interactive_symbol, hedge_used
        nonlocal stock_edits, composition_attempted, regression_repair
        nonlocal alternate, test_seed
        interactive_target = interactive_symbol = None
        interactive_misses.clear()
        unlocalized_plans.clear()
        hedge_used = False
        continuations.clear()
        stock_edits, composition_attempted = (), False
        regression_repair = None
        alternate = None
        test_seed = None
        attempted_seed_cases.clear()
        pinned_restore_advice.clear()
        repair_attempted.clear()
        unavailable.clear()
        rejected.clear()
        planning_attempts.clear()
        model_failures.clear()
        retry_advice.clear()
        line_retries.clear()
        line_routes.clear()
        mock_diagnostics.clear()
        round_limit_advice.clear()
        counter_sites.clear()
        last_routes.pop(target, None)
        notes.pop(target, None)
        malformed.pop(target, None)
        native_ready.discard(target)
        no_progress.pop(target, None)

    if (durable_witness and focused_generated and executor.verification_runner is not executor.runner
            and getattr(executor.verification_runner, "network", "none") == "none"
            and budget.commands_remaining() >= len(commands) * 2 + 12 and budget.remaining() >= 75):
        try:
            test_first_witness = prove_red_regression(durable_witness.source)
        except Cancelled:
            raise
        except (ExecutionError, OSError, ValueError):
            budget.remaining()
        if test_first_witness:
            test_first_attempted = True
            _LOG.info("Reverified saved regression assertion failures on both revisions")
        else:
            durable_witness = None
            retain_recovery(durable=True)

    pool = ThreadPoolExecutor(max_workers=executor.limits.concurrent_probes)
    try:
        while True:
            if retired is not None and retired.done():
                retired = None
            needs_tests = task.require_new_tests and not has_runnable_test(
                new_path, current[new_path].encode()
            )
            if test_first_witness and not failures and needs_tests and not pending:
                retained, _ = test_first_witness
                test_first_witness = None
                combined = patch_for(original, {**current, new_path: retained})
                if len(combined) <= executor.limits.patch_bytes:
                    try:
                        return _finish_review(executor, task, archive, budget, combined, actor, version)
                    except AttemptRejected as error:
                        if not isinstance(error.__cause__, _VerificationRejected):
                            raise
                        _LOG.info("Test-first candidate failed final verification; resuming standard test authoring")
            ready = not failures and not needs_tests and bool(patch)
            if ready and not pending:
                return _finish_review(
                    executor, task, archive, budget, patch, actor, version
                )
            # Check diagnoses and test authoring retain single-request behavior.
            operator = [key for key in failures if key[0] < len(task.pull_checks)]
            other_checks = [key for key in failures if key not in operator]
            remaining = (
                operator + other_checks[:1]
                if operator
                else other_checks or ([_TEST_GOAL] if needs_tests else [])
            )
            if not remaining and not ready:
                return None
            limit = executor.limits.concurrent_probes if operator else 1
            active = {request[0] for request in pending.values()}
            private = regression_repair or next(
                (
                    request[-1]
                    for request in pending.values()
                    if request[1] is current and request[-1] is not None
                ),
                None,
            )
            repair_target = (
                regression_repair.next_target(commands) if regression_repair else None
            )
            available = sorted(
                (
                    key
                    for key in remaining
                    if key not in active
                    and key not in unavailable
                    and (
                        private is None
                        or (key != private.origin and key not in private.gained)
                    )
                ),
                # A failing parameter identifies a concrete input in a family of cases.
                key=lambda key: (
                    key not in continuations,
                    attempts.get(key, 0),
                    "[" not in key[1],
                    key,
                ),
            )
            if repair_target is not None:
                available.insert(0, repair_target)
            launch_budget = min(
                max_proposals - submitted, budget.model_requests_remaining()
            )
            if (
                not test_first_attempted and current is resumed_operator_checkout and needs_tests
                and len(operator) == 1 and not other_checks and not pending and not private
                and _named_operator_cases(operator, commands)
                and any(run and run.passed and not run.deselected for run in pytest_runs.values())
                and attempts.get(operator[0], 0) >= 1 and launch_budget >= 4 and focused_generated
                and executor.verification_runner is not executor.runner
                and getattr(executor.verification_runner, "network", "none") == "none"
                and budget.commands_remaining() >= len(commands) * 2 + 12 and budget.remaining() >= 75
            ):
                test_first_attempted = True
                routes = []
                after_provider = None
                runnable_drafts = {}
                for attempt in (1, 2, 3):
                    if attempt == 3 and not runnable_drafts and not test_first_runtime:
                        break
                    _LOG.info("Requesting test-first regression authoring, attempt %d", attempt)
                    submitted += 1
                    launch_budget -= 1
                    try:
                        test_first_witness = author_red_regression(routes, after_provider, runnable_drafts)
                        after_provider = routes[-1][0] if routes else None
                    except Cancelled:
                        raise
                    except ProviderExhausted as error:
                        # A new account is useful after a deadline interrupted gateway
                        # rotation. A completed exhaustion already visited eligible routes.
                        after_provider = None
                        if error.failures.get("gateway") == "deadline exceeded":
                            after_provider = next((alias for alias, reason in reversed(error.failures.items())
                                if alias != "gateway" and reason in {"upstream timeout", "upstream connection failed"}), None)
                        _LOG.info("Test-first provider request exhausted; alternate available=%s", bool(after_provider))
                    except (ExecutionError, OSError, ValueError):
                        _LOG.info("Test-first attempt ended without a verified regression")
                        budget.remaining()
                        after_provider = routes[-1][0] if routes else None
                    if (
                        test_first_witness or not after_provider or launch_budget < 3
                        or budget.model_requests_remaining() < 3 or budget.remaining() < 75
                        or budget.commands_remaining() < len(commands) * 2 + 12
                    ):
                        break
                if test_first_witness:
                    durable_witness = RedWitness(witness_identity(), test_first_witness[0])
                    retain_recovery(durable=True)
                    _LOG.info("Retained an authored regression with verified assertion failures on both revisions")
            if (
                len(operator) == 1
                and not other_checks
                and not pending
                and retired is None
                and private is None
                and alternate is None
                and patch
                and actor is not None
                and actor.role == "coding"
                and actor.configured_model
                and executor.verification_runner is not executor.runner
                and needs_tests
                and not current[new_path]
                and launch_budget >= 1
                and round_limit_snapshot is not current
            ):
                round_limit_snapshot = current
                target = operator[0]
                verified, advisory = round_limit_trial(target)
                if verified is not None:
                    current, patch, failures, pytest_runs = verified
                    start_new_draft(target)
                    save_verified_progress(
                        patch, "Round-limit change passed all configured checks. New tests are pending.", actor
                    )
                    _LOG.info("A private round-limit trial passed all configured checks")
                    continue
                if advisory:
                    round_limit_advice[target] = advisory
            if (
                not mixed_guard_attempted
                and operator
                and len(other_checks) == 1
                and not pending
                and retired is None
                and private is None
                and alternate is None
                and patch
                and actor is not None
                and actor.role == "coding"
                and actor.configured_model
                and executor.verification_runner is not executor.runner
                and launch_budget >= 1 + int(needs_tests)
                and budget.commands_remaining() >= 3 * len(commands) + 6
                and budget.remaining() >= 150
            ):
                mixed_guard_attempted = True
                stock = other_checks[0]
                verified, failed_trial = guard_order_with_operator_failures(
                    stock, set(operator)
                )
                if verified is not None:
                    current, patch, failures, pytest_runs = verified
                    start_new_draft(stock)
                    _LOG.info(
                        "An adjacent stock filter reorder passed without new review failures"
                    )
                    continue
                if failed_trial and (
                    try_semantic_record_key(stock, *failed_trial, mixed=True)
                    or try_guard_precision(stock, *failed_trial, verify=verify_mixed_stock_gain)
                ):
                    continue
                if failed_trial and (
                    site := stock_failure_site(stock[1], failed_trial[1])
                ):
                    notes[stock] = (
                        f"The trial guard reorder still failed this case at {site}."
                    )
            precision_ready = (
                eager_target is not None
                and not precision_attempted
                and executor.verification_runner is not executor.runner
                and launch_budget >= 1 + int(needs_tests)
                and budget.commands_remaining() >= 3 * len(commands) + 4
                and budget.remaining() >= _LOOP_PRECISION_MIN_REMAINING
            )
            if (
                eager_target is not None
                and current is eager_source
                and failures.keys() == {eager_target}
                and eager_trace is None
                and (not pinned_restore_attempted or precision_ready)
                and patch
                and not pending
                and retired is None
                and private is None
                and (alternate is None or pinned_restore_attempted)
                and launch_budget >= 1 + int(needs_tests)
                and budget.commands_remaining() >= 3 * len(commands) + 6
                and budget.remaining() >= 115
            ):
                eager_trace = ""
                command = commands[eager_target[0]]
                diagnostic = failures[eager_target]
                fixture = stock_pytest_excerpt(
                    archive, command, eager_target[1], diagnostic
                )
                stock_fixtures[eager_target, diagnostic] = fixture
                if fixture:
                    eager_trace = mock_call_diagnostic(
                        executor,
                        task,
                        archive,
                        budget,
                        command,
                        eager_target[1],
                        patch,
                        failure_line=stock_assertion_failure_line(
                            eager_target[1], diagnostic
                        ),
                        reviewed_sources=(original, current),
                    )
                    loop = counter_loop_reuse(eager_trace)
                    if (
                        loop is not None
                        and needs_tests
                        and not current.get(new_path)
                        and loop.node
                        == eager_target[1]
                        .split("[", 1)[0]
                        .removeprefix("/work/")
                        .removeprefix("./")
                        and loop.assertion_line
                        == stock_assertion_failure_line(eager_target[1], diagnostic)
                        and budget.remaining() >= 90
                    ):
                        allowed = {
                            path
                            for path in paths
                            if not _protected(path, task, new_test=False)
                        }
                        reordered = swap_traced_filters(current, allowed, loop)
                        if reordered is not None:
                            verified, focused_failure = verify_guard_order(
                                eager_target, reordered
                            )
                            if verified is not None:
                                current, patch, failures, pytest_runs = verified
                                start_new_draft(eager_target)
                                _LOG.info(
                                    "An adjacent stock filter reorder passed all configured checks"
                                )
                                continue
                            if try_semantic_record_key(
                                eager_target, eager_trace, focused_failure
                            ) or try_guard_precision(eager_target, eager_trace, focused_failure):
                                continue
                            launch_budget = min(
                                max_proposals - submitted,
                                budget.model_requests_remaining(),
                            )
                    located = pinned_counter_location(
                        original, current, counter_caller_site(eager_trace)
                    )
                    if (
                        not pinned_restore_attempted
                        and located is not None
                        and located.current != located.baseline
                        and len(located.current.encode())
                        >= _LONG_LOCALIZED_DEFINITION_BYTES
                        and launch_budget >= 1 + int(needs_tests)
                        and budget.remaining() >= 75
                    ):
                        pinned_restore_attempted = True
                        regression_repair, eager_advisory = probe_pinned_restore(
                            eager_target, located
                        )
                        if regression_repair is not None:
                            repair_attempted.add(eager_target)
                            continuations.pop(eager_target, None)
                            _LOG.info("Verified pinned definition restore entered private stock repair")
                            continue
                        if eager_advisory:
                            mock_diagnostics[eager_target] = eager_advisory
            if (
                eager_target is not None
                and current is eager_source
                and failures.keys() == {eager_target}
                and precision_ready
                and not precision_attempted
                and (signal := counter_loop_reuse(eager_trace)) is not None
                and not pending
                and retired is None
                and private is None
                and (alternate is None or pinned_restore_attempted)
                and budget.commands_remaining() >= 3 * len(commands) + 4
                and budget.remaining() >= _LOOP_PRECISION_MIN_REMAINING
            ):
                precision_attempted = True
                before = budget.model_requests_remaining()
                proposed = precision_stock_gain(eager_target, signal)
                submitted += before - budget.model_requests_remaining()
                if proposed is not None:
                    (current, patch, failures, pytest_runs), note, actor = proposed
                    editor = actor
                    start_new_draft(eager_target)
                    if failures or needs_tests:
                        save_verified_progress(patch, note, editor)
                    continue
                launch_budget = min(max_proposals - submitted, budget.model_requests_remaining())
            if interactive_target is None and len(failures) == 1:
                sole_target = next(iter(failures))
                # Let the ranked planner try a verified plateau first. If it
                # cannot localize it, the isolated tools may inspect the files.
                if (
                    interactive_misses[sole_target] >= RECOVERY_MODE_THRESHOLD
                    and (
                        sole_target in unlocalized_plans
                        or no_progress.get(sole_target, 0) < RECOVERY_MODE_THRESHOLD
                    )
                ):
                    interactive_target, interactive_symbol = sole_target, None
            if (
                interactive_target is not None
                and interactive_symbol is not None
                and localized_retry_symbol(pinned_restore_advice.get(interactive_target))
                == interactive_symbol
                and not pinned_restore_attempted
                and not pending
                and retired is None
                and private is None
                and alternate is None
                and failures.keys() == {interactive_target}
                and patch
                and actor is not None
                and actor.role == "coding"
                and actor.configured_model
                and interactive_target not in line_retries
                and interactive_target not in retry_advice
                and (stock_run := pytest_runs.get(interactive_target[0])) is not None
                and stock_run.failed == {interactive_target[1]}
                and launch_budget >= 1 + int(needs_tests)
                and budget.commands_remaining() >= 3 * len(commands) + 4
                and budget.remaining() >= 75
            ):
                pinned_restore_attempted = True
                regression_repair, dependency_advisory = probe_pinned_restore(
                    interactive_target, pinned_restore_advice[interactive_target]
                )
                if dependency_advisory:
                    mock_diagnostics[interactive_target] = "\n".join(
                        filter(
                            None,
                            (mock_diagnostics.get(interactive_target), dependency_advisory),
                        )
                    )
                if regression_repair is not None:
                    repair_attempted.add(interactive_target)
                    continuations.pop(interactive_target, None)
                    unavailable.clear()
                    _LOG.info("Verified pinned definition restore entered private stock repair")
                    continue
            if (
                interactive_target is not None
                and not interactive_attempted
                and not pending
                and retired is None
                and private is None
                and patch
                and failures.keys() == {interactive_target}
                and stock_fixtures.get((interactive_target, failures[interactive_target]))
                and (stock_run := pytest_runs.get(interactive_target[0])) is not None
                and stock_run.failed == {interactive_target[1]}
                and _verified_operator_runs(
                    pytest_runs, pytest_runs, set(), task.pull_checks, operator_cases
                )
                and interactive_target not in line_retries
                and interactive_target not in retry_advice
                # Keep any unverified continuation available if tool-assisted repair fails.
                and retained_recovery is not None
                and retained_recovery.identity == retained_identity
                and any(
                    goal == _goal_digest(interactive_target)
                    and max(bad, stalled) >= RECOVERY_MODE_THRESHOLD
                    for goal, bad, stalled in retained_recovery.goals
                )
                and executor.verification_runner is not executor.runner
            ):
                turns = min(
                    budget.limits.rollout_requests or _INTERACTIVE_MAX_TURNS,
                    _INTERACTIVE_MAX_TURNS,
                    launch_budget - int(needs_tests),
                )
                tools = min(_INTERACTIVE_MAX_TOOLS, budget.commands_remaining() - 2 * len(commands) - 1)
                if turns >= 5 and tools >= 6 and budget.remaining() >= _INTERACTIVE_RESERVE_SECONDS + 90:
                    interactive_attempted = True
                    before = budget.model_requests_remaining()
                    _LOG.info("Trying isolated interactive repair for %s", interactive_target[1])
                    proposed = interactive_stock_gain(interactive_target, interactive_symbol, _InteractiveBudget(budget, turns, tools))
                    submitted += before - budget.model_requests_remaining()
                    if proposed is not None:
                        (
                            (current, patch, failures, pytest_runs),
                            actor,
                            progress_note,
                            tentative_tests,
                        ) = proposed
                        editor = actor
                        start_new_draft(interactive_target)
                        if progress_note:
                            _LOG.info("%s", progress_note)
                        if failures or task.require_new_tests:
                            save_verified_progress(patch, progress_note, editor)
                        # Speculative tests are never written into this checkpoint.
                        if task.require_new_tests and tentative_tests:
                            test_draft = tentative_tests
                            if not failures and has_runnable_test(
                                new_path, test_draft.encode()
                            ):
                                with_tests = {**current, new_path: test_draft}
                                combined = patch_for(original, with_tests)
                                if len(combined) <= executor.limits.patch_bytes:
                                    current, patch = with_tests, combined
                        continue
                    launch_budget = min(
                        max_proposals - submitted, budget.model_requests_remaining()
                    )
            if (
                stock_hedge is not None
                and len(pending) == 1
                and retired is None
                and private is None
                and launch_budget > 0
                and not stock_hedge.primary.done()
                and pending[stock_hedge.primary][1] is current
                and time.monotonic() >= stock_hedge.due
                and budget.remaining() > 30
            ):
                primary = stock_hedge.primary
                alternate = pool.submit(
                    _coding_completion,
                    executor,
                    budget,
                    stock_hedge.messages,
                    stock_hedge.route,
                    None,
                    exclude=stock_hedge.exclude,
                )
                pending[alternate] = pending[primary]
                hedge_pair = {primary, alternate}
                stock_hedge, hedge_used = None, True
                submitted += 1
                launch_budget -= 1
            # A running sibling still occupies its worker after it becomes obsolete.
            capacity = executor.limits.concurrent_probes - len(pending) - int(retired is not None)
            slots = min(launch_budget, limit - len(pending), capacity)
            if not ready and slots > 0:
                for target in available[:slots]:
                    if not launch_budget:
                        break
                    attempts[target] = attempts.get(target, 0) + 1
                    repair = regression_repair if target == repair_target else None
                    if repair is None and target in operator and (
                        malformed.get(target) or current is resumed_operator_checkout
                    ):
                        if malformed.get(target):
                            queue_iteration_retry(target, current)
                        if target not in line_retries:
                            queue_review_anchored_retry(target)
                    line_retry = line_retries.pop(target, None)
                    if (
                        line_retry is not None and line_retry.checkout is not None
                        and (line_retry.checkout is not current or repair is not None)
                    ):
                        line_retry = None
                    if repair is not None and repair.private_symbol and target in repair.introduced:
                        line_retry = _private_method_retry(current, repair, target, review)
                    if (
                        production_regression and repair is None and target in operator
                        and not (current is resumed_operator_checkout and line_retry is not None)
                        and launch_budget >= 2
                        and (
                            attempts[target] >= 5
                            or (target in restored_raw_targets and patch == restored_raw_patch)
                        )
                        and malformed.get(target, 0) >= RECOVERY_MODE_THRESHOLD
                        and (observed := pytest_runs.get(target[0])) is not None
                        and target[1] in observed.failed
                        and (prior_raw := raw_nested_attempts.get(hashlib.sha256(patch).digest(), ())) is not None
                        and len(prior_raw) < 3
                    ):
                        revision = hashlib.sha256(patch).digest()
                        # Close this slot while the request runs. Failed formats or
                        # unchanged checks can reopen it for a different coding model.
                        raw_nested_attempts[revision] = None
                        raw_retry = _raw_nested_messages(
                            _draft_review(review, original, current), current,
                            task.regression_edit_path, target[1], failures[target],
                        )
                        if raw_retry is not None:
                            if line_retry is not None:
                                line_retries[target] = line_retry
                            symbol, messages = raw_retry
                            future = pool.submit(
                                _coding_completion, executor, budget, messages,
                                last_routes.get(target), None, exclude=prior_raw, raw_python=True,
                            )
                            pending[future] = (
                                target, current, None, False, None, (None, False),
                                None, None, None, False, None, (symbol, revision, prior_raw), None,
                            )
                            submitted += 1
                            launch_budget -= 1
                            continue
                    scoped_line = (
                        line_retry is not None and line_retry.allowed_rows is not None
                    )
                    native_line = (
                        scoped_line and line_retry.native_ready and not line_retry.native_attempted
                        and launch_budget >= 2
                    )
                    if native_line:
                        line_retry = replace(line_retry, native_attempted=True)
                    advice = (
                        line_retry.advice
                        if line_retry
                        else retry_advice.pop(target, None)
                    )
                    # Give the strategy one more attempt on its original draft.
                    # Keep any unpublished continuation available as a fallback.
                    continuation = (
                        continuations.pop(target, None) if advice is None else None
                    )
                    if continuation is None:
                        source, diagnostics = current, failures
                        note = notes.get(target, review.previous_failure)
                    else:
                        source, diagnostic = continuation
                        diagnostics = {**failures, target: diagnostic}
                        note = (
                            "This checkout includes one unpublished proposal. The same "
                            "assertion still fails. Continue from these edits once."
                        )
                    if repair is not None:
                        regression_repair = None
                        source, diagnostics = repair.sources, repair.failures
                        note = repair.retry_note if repair.retry_target == target else ""
                        _LOG.info(
                            "Verifying private repair of %d review case(s) and %d static check(s)",
                            len(repair.introduced),
                            len(repair.static_failures),
                        )
                    allow_test_module = False
                    native_test = False
                    test_line_snapshot = None
                    test_raw_retry = None
                    helper_authoring = None
                    helper_messages = None
                    if target == _TEST_GOAL:
                        if test_checkout_snapshot is not current:
                            test_checkout_snapshot = current
                            test_checkout_digest = hashlib.sha256(patch).digest()
                            test_case_runs = {}
                            test_passed_ancestor = False
                            test_setup_failures = 0
                            prior_test_attempted = False
                            prior_test_source = ""
                            prior_test_fallback = None
                            helper_setup = None
                            helper_attempts = 0
                            helper_outcomes = Counter()
                            helper_setup_misses = 0
                            helper_feedback = ""
                            helper_function = ""
                            helper_bad_model = ()
                            test_model_failures = Counter()
                        if repository_example is None or (pinned_ancestor and repository_checkout is not current):
                            repository_checkout = current
                            reviewed_sources = {path: current[path] for path in paths}
                            repository_example, repository_usage = repository_pytest_context(
                                task.regression_archive or archive,
                                reviewed_sources, " ".join(body for _, _, body in review.feedback),
                                exclude_generated=production_regression,
                            )
                            repository_models = imported_model_declarations(archive, reviewed_sources)
                            if pinned_ancestor:
                                public_sources = {
                                    path: text for path, text in reviewed_sources.items()
                                    if not _protected(path, task, new_test=False)
                                }
                                reviewed_feedback = _draft_review(review, original, current).feedback
                                ancestor_delta = reviewed_ancestor_delta(
                                    task.regression_archive, public_sources, reviewed_feedback,
                                )
                                extended_context = extended_pytest_context(
                                    task.regression_archive, public_sources, reviewed_feedback,
                                )
                                prior_setup = prior_generated_fixture_context(
                                    archive, task.regression_archive, new_path, public_sources,
                                    require_identical=production_regression,
                                ) if review.test_only or production_regression else ""
                                if prior_setup and (production_regression or len(prior_setup.encode()) <= 850):
                                    repository_example = prior_setup
                                elif prior_setup:
                                    extended_context = "\n\n".join(filter(None, (extended_context, prior_setup)))
                        allow_test_module = bool(
                            original[new_path] == current[new_path] == ""
                            and not _protected(new_path, task, new_test=True)
                            and (not test_draft or test_diagnostic)
                        )
                        if (
                            production_regression and allow_test_module and focused_generated
                            and test_setup_failures >= 2 and not prior_test_attempted
                            and not ((run := test_case_runs.get("focused")) and run.passed)
                        ):
                            prior_test_attempted = True
                            if seed := verified_prior_seed():
                                path, name, prior_test_source, run = seed
                                prior_test_fallback = (
                                    test_draft, test_diagnostic, test_ancestor_feedback,
                                    test_rejection, test_case_runs, test_passed_ancestor,
                                )
                                test_draft = prior_test_source
                                test_case_runs = {"focused": run}
                                test_passed_ancestor = True
                                test_rejection = None
                                test_diagnostic = (
                                    f"This editable predecessor seed from {path} passed on both revisions. "
                                    f"Keep its working setup and strengthen {new_path}::{name} to assert the "
                                    "requested behavior. Prefer one named replacement of that test function."
                                )
                                test_ancestor_feedback = (
                                    "The editable predecessor seed also passed on the pinned ancestor. "
                                    "Author an assertion that passes here and fails there."
                                )
                                notes.pop(target, None)
                                _LOG.info("Started new pytest authoring from the verified predecessor seed")
                        revision = (test_checkout_digest, hashlib.sha256(test_draft.encode()).digest())
                        if revision != test_format_revision:
                            test_format_revision = revision
                            test_format_failures = 0
                        smoke = lone_both_pass_case()
                        if (
                            production_regression and allow_test_module and fixtures and test_seed is None
                            and helper_attempts < 8 and helper_setup_misses < 2
                            and max(helper_outcomes.values(), default=0) < 3
                            and (not helper_attempts or helper_feedback or helper_bad_model)
                            and (
                                smoke or helper_feedback
                                or current is resumed_verified_checkout and not test_draft and not test_case_runs
                                or test_setup_failures >= 2
                                and not ((run := test_case_runs.get("focused")) and run.passed)
                            )
                            and _one_test_draft(test_draft)
                        ):
                            if helper_setup is None:
                                helper_setup = operator_test_assembly(
                                    fixtures, archive, task.regression_archive,
                                    {path: current[path] for path in paths if not _protected(path, task, new_test=False)},
                                ) or False
                            if helper_setup:
                                offered = _operator_function_messages(
                                    helper_setup, review, current, new_path, ancestor_delta,
                                    replace_smoke=bool(smoke), feedback=helper_feedback, prior_function=helper_function,
                                )
                                if wire_size(offered) <= 12 * 1024:
                                    helper_authoring = helper_setup
                                    helper_messages = offered
                                    helper_attempts += 1
                                else:
                                    helper_setup = False
                        native_test = (
                            allow_test_module and helper_authoring is None
                            and test_format_failures >= 2
                            and revision not in test_tool_attempts
                            and launch_budget >= 2
                        )
                        if native_test:
                            # Each stored digest pair costs a reserved proposal.
                            test_tool_attempts.add(revision)
                            _LOG.info("Trying structured pytest module reply after repeated format errors")
                        elif (
                            allow_test_module and helper_authoring is None
                            and test_format_failures > RECOVERY_MODE_THRESHOLD
                            and (prior := test_raw_attempts.get(revision, ())) is not None
                            and len(prior) < 2
                            and (raw := _failed_test_line_snapshot(
                                new_path, test_draft, test_diagnostic, test_case_runs.get("focused"), raw=True,
                            ))
                        ):
                            test_raw_attempts[revision] = None
                            test_raw_retry = _RawTestRetry(raw, revision, prior)
                        elif (
                            allow_test_module and helper_authoring is None
                            and test_format_failures and revision not in test_line_attempts
                        ):
                            test_line_snapshot = _failed_test_line_snapshot(
                                new_path, test_draft, test_diagnostic, test_case_runs.get("focused"),
                            )
                            if test_line_snapshot is not None:
                                test_line_attempts.add(revision)
                        source = {**current, new_path: test_draft}
                        note = "\n".join(
                            item
                            for item in (test_diagnostic, notes.get(target))
                            if item
                        )
                    check_command = commands[target[0]] if target in diagnostics else ()
                    stock_fixture = None
                    if (
                        target[0] >= len(task.pull_checks)
                        and check_command
                        and check_label(check_command, ()) == "pytest"
                    ):
                        cache_key = target, diagnostics[target]
                        if cache_key not in stock_fixtures:
                            stock_fixtures[cache_key] = stock_pytest_excerpt(
                                archive, check_command, target[1], diagnostics[target]
                            )
                        stock_fixture = stock_fixtures[cache_key]
                    can_plan = target in operator or bool(stock_fixture)
                    named_definitions = (
                        target in failures
                        and (
                            malformed.get(target, 0) >= RECOVERY_MODE_THRESHOLD
                            or (
                                can_plan
                                and continuation is None
                                and repair is None
                                and no_progress.get(target, 0)
                                >= RECOVERY_MODE_THRESHOLD
                                and attempts[target] >= 6
                                and attempts[target] % 3 == 0
                            )
                        )
                        # Import blocks cannot be replaced by naming a definition.
                        and (
                            not check_command
                            or check_label(check_command, ()) != "ruff"
                        )
                    )
                    if repair is not None and repair.stock_origin:
                        named_definitions = (
                            repair.retry_named
                            and repair.retry_target == target
                            and check_label(check_command, ()) != "ruff"
                        )
                    localized_retry = (
                        localized_retry_symbol(advice)
                        if stock_fixture and repair is None
                        else None
                    )
                    numbered = (
                        prepare_line_range_edit(source, localized_retry)
                        if localized_retry is not None
                        else None
                    )
                    if (
                        line_retry is not None and not scoped_line
                        and numbered != line_retry.snapshot
                    ):
                        line_retry, advice, localized_retry = None, None, None
                    line_followup = (
                        _LineRetry(advice, numbered)
                        if numbered is not None and line_retry is None and advice is not None
                        else None
                    )
                    native = (
                        named_definitions
                        and not scoped_line
                        and localized_retry is None
                        and target in native_ready
                        and target not in native_attempted
                        and launch_budget >= 2
                        and not (repair is not None and repair.stock_origin)
                    )
                    if native:
                        native_attempted.add(target)
                    plan_count, last_planned = planning_attempts.get(target, (0, -3))
                    plan = (
                        can_plan
                        and not scoped_line
                        and not native
                        and target not in round_limit_advice
                        and advice is None
                        and repair is None
                        and plan_count < 2
                        and attempts[target] - last_planned >= 3
                        and no_progress.get(target, 0) >= RECOVERY_MODE_THRESHOLD
                        and continuation is None
                        and launch_budget >= 2
                        and budget.remaining() > 30
                    )
                    if plan and stock_fixture and patch and target not in counter_sites:
                        cached_trace = (
                            eager_trace
                            if target == eager_target and source is eager_source
                            else None
                        )
                        traces = cached_trace if cached_trace is not None else mock_call_diagnostic(
                            executor, task, archive, budget, check_command, target[1], patch,
                            failure_line=stock_assertion_failure_line(
                                target[1], diagnostics[target]
                            ),
                            reviewed_sources=(original, source),
                        )
                        counterfactual = stock_guard_diagnostic(
                            executor, task, archive, budget, check_command, target[1],
                            original, source, stock_fixture, patch_for,
                        )
                        mock_diagnostics[target] = "\n".join(
                            filter(None, (
                                counterfactual, traces, eager_advisory if cached_trace is not None else "",
                            ))
                        )
                        counter_sites[target] = counter_caller_site(traces)
                        if counterfactual:
                            _LOG.info("Isolated guard replay localized the tracked pytest regression")
                    mock_detail = mock_diagnostics.get(target, "")
                    if continuation is not None and mock_detail:
                        tentative_trace = mock_call_diagnostic(
                            executor, task, archive, budget, check_command, target[1],
                            patch_for(original, source),
                            failure_line=stock_assertion_failure_line(
                                target[1], diagnostics[target]
                            ),
                            reviewed_sources=(original, source),
                        )
                        saved_trace = "Observations from before the unpublished edit:\n" + mock_detail
                        mock_detail = (
                            "Observed calls on the unpublished source being edited now:\n" + tentative_trace
                            + "\n" + excerpt(saved_trace, 1300)
                            if tentative_trace else saved_trace
                        )
                        _LOG.info("Continuing an unpublished edit for tracked pytest case %s", target[1])
                    if (
                        not plan
                        and repair is None
                        and source is current
                        and (advisory := round_limit_advice.pop(target, ""))
                    ):
                        mock_detail = "\n".join(
                            filter(
                                None,
                                (
                                    mock_detail,
                                    "Verified draft source observation (quoted): "
                                    + advisory,
                                ),
                            )
                        )
                    # Begin with SEARCH. A localized retry can replace the one
                    # complete method only when both exact definitions fit.
                    if stock_fixture and (plan or advice is not None):
                        named_definitions = localized_retry is not None
                    parameter_focus = ""
                    if target in operator:
                        parameter_note = lost_iteration_argument(
                            _fixture_excerpt(fixtures, target[1], selected_only=True),
                            original,
                            source,
                            source_windows=(
                                repair is None
                                and source is current
                                and _trusted_operator_node(
                                    target[1], check_command, operator_cases
                                )
                            ),
                        )
                        if "\nCurrent traversal " in parameter_note:
                            parameter_focus = parameter_note
                        elif parameter_note:
                            mock_detail = "\n".join(filter(None, (mock_detail, parameter_note)))
                    messages = helper_messages or _proposal_messages(
                        target,
                        source,
                        diagnostics,
                        needs_tests,
                        note,
                        _draft_review(review, original, source),
                        fixtures,
                        (
                            [repository_models, *interfaces]
                            if repository_models and target == _TEST_GOAL
                            else interfaces
                        ),
                        new_path,
                        continuing=continuation is not None or repair is not None,
                        named_definitions=named_definitions,
                        check_command=check_command,
                        stock_fixture=stock_fixture,
                        preserved_stock_fixture=(
                            stock_fixtures.get(
                                (repair.origin, failures.get(repair.origin)), ""
                            )
                            if repair is not None and repair.stock_origin
                            else ""
                        ),
                        private_source=(
                            private_operator_source(
                                current, source, fixtures, target[1]
                            )
                            if repair is not None
                            and repair.stock_origin
                            and target in repair.introduced
                            else ""
                        ),
                        mock_detail=mock_detail,
                        parameter_focus=parameter_focus,
                        allow_test_module=allow_test_module,
                        repository_example=repository_example or "",
                        repository_usage=repository_usage,
                        repair_failed_tests=target == _TEST_GOAL and bool(test_draft and test_diagnostic),
                        prefer_test_module=bool(
                            target == _TEST_GOAL and review.test_only
                            and (test_run := test_case_runs.get("focused"))
                            and test_run.named_cases and not test_run.passed and not test_run.deselected
                        ),
                        ancestor_evidence=(
                            (test_ancestor_feedback or
                             "Write an assertion that passes here and fails on the pinned ancestor.")
                            + ("\n" + ancestor_delta if ancestor_delta else "")
                            if target == _TEST_GOAL and pinned_ancestor else ""
                        ),
                        extended_context=extended_context,
                        rejection=(
                            _private_repair_context(current, repair)
                            if repair is not None
                            else rejected.get(target, "")
                            if continuation is None
                            else ""
                        ),
                    )
                    if test_raw_retry is not None:
                        messages = _test_line_messages(
                            test_raw_retry.snapshot, note, source, repository_usage=repository_usage, raw=True,
                        )
                    elif test_line_snapshot is not None:
                        messages = _test_line_messages(
                            test_line_snapshot, note, source, repository_usage=repository_usage,
                        )
                    elif scoped_line:
                        messages = [
                            {"role": "system", "content": _LINE_RANGE_SYSTEM},
                            {"role": "user", "content": line_retry.prompt},
                        ]
                    if test_first_witness and (target in operator or repair is not None and target in repair.introduced):
                        hint = test_first_witness[1] + "\n"
                        augmented = [*messages[:-1], {**messages[-1], "content": hint + messages[-1]["content"]}]
                        if wire_size(augmented) <= RETRY_MESSAGE_BYTES:
                            messages = augmented
                            if native_line and line_retry:
                                line_retry = replace(line_retry, native_prompt=hint + line_retry.native_prompt)
                    plan_private_method = bool(
                        repair and repair.private_symbol and repair.stalled_edits and repair.excluded_models
                        and scoped_line and line_retry.complete_single_method and not native_line
                        and launch_budget >= 3 and budget.remaining() >= 90
                        and (target, line_retry.snapshot.digest) not in planned_private_methods
                    )
                    if plan_private_method:
                        planned_private_methods.add((target, line_retry.snapshot.digest))
                        _LOG.info("Requesting behavioral planning advice for a verified private method")
                    planner_input = None
                    if plan:
                        fixture = stock_fixture or _fixture_excerpt(
                            fixtures, target[1], selected_first=True
                        )
                        planner_input = (
                            source, fixture, diagnostics[target], mock_detail,
                            (original, fixtures, review.feedback) if stock_fixture else None,
                            counter_sites.get(target),
                        )
                        planning_attempts[target] = plan_count + 1, attempts[target]
                    rotation_key = None
                    exclude = repair.excluded_models if repair is not None else ()
                    unassisted_text = (
                        target != _TEST_GOAL
                        and repair is None
                        and planner_input is None
                        and advice is None
                        and not native
                    )
                    if unassisted_text and continuation is None:
                        rotation_key = (target, named_definitions)
                        counts = model_failures.setdefault(rotation_key, Counter())
                        if counts is not None:
                            exclude = tuple(
                                model for model, count in counts.items()
                                if count >= _MODEL_FAILURE_THRESHOLD
                            )
                    if target == _TEST_GOAL and test_model_failures:
                        exclude = tuple(dict.fromkeys(
                            pair for (pair, _), count in test_model_failures.items()
                            if count >= _MODEL_FAILURE_THRESHOLD
                        ))
                    if test_raw_retry is not None:
                        exclude = tuple(dict.fromkeys((*exclude, *test_raw_retry.excluded_models)))
                    if helper_authoring is not None and helper_bad_model:
                        exclude = tuple(dict.fromkeys((*exclude, helper_bad_model)))
                    if line_retry is not None and line_retry.excluded_models:
                        exclude = tuple(dict.fromkeys((*exclude, *line_retry.excluded_models)))
                    route = (
                        line_routes.get(target)
                        if line_retry and not scoped_line else last_routes.get(target)
                    )
                    advice_source = "planning" if planner_input is not None else "guided" if advice is not None else None
                    if repair is not None and repair.private_symbol and line_retry is not None:
                        _LOG.info("Requesting private method repair for %s (attempt %d)",
                                  repair.private_symbol[1], repair.retries + 1)
                    elif line_retry is not None and line_retry.direct_caller_only:
                        _LOG.info(
                            "Requesting checkpoint method edit: direct caller only %s (body rows=%d)",
                            line_retry.snapshot.symbol[1], len(line_retry.allowed_rows),
                        )
                    elif line_retry is not None and line_retry.checkpoint_method_edit:
                        caller_rows = ", ".join(
                            f"{candidate.symbol[1]} (body rows={len(candidate.lines) - candidate.first_body_line + 1})"
                            for candidate in line_retry.alternatives
                        )
                        _LOG.info(
                            "Requesting checkpoint method edit: anchor %s (body rows=%d), direct callers offered (%d): %s",
                            line_retry.snapshot.symbol[1], len(line_retry.allowed_rows),
                            len(line_retry.alternatives), caller_rows or "none",
                        )
                    launched_at = time.monotonic()
                    future = pool.submit(
                        _coding_completion,
                        executor,
                        budget,
                        messages,
                        route,
                        planner_input,
                        source if native else None,
                        advice,
                        exclude=exclude,
                        line_snapshot=line_retry.snapshot if line_retry else None,
                        scoped_line=scoped_line,
                        native_test_path=new_path if native_test else None,
                        native_line_retry=line_retry if native_line else None,
                        plan_private_method=plan_private_method,
                    )
                    pending[future] = (
                        target,
                        current,
                        continuation,
                        named_definitions,
                        advice_source,
                        (rotation_key, bool(exclude)),
                        localized_retry,
                        line_retry,
                        line_followup,
                        allow_test_module,
                        helper_authoring or test_raw_retry or test_line_snapshot,
                        None,
                        repair,
                    )
                    if (
                        not hedge_used
                        and executor.limits.concurrent_probes >= 2
                        and stock_fixture
                        and failures.keys() == {target}
                        and unassisted_text
                        and retained_recovery is not None
                        and retained_recovery.identity == retained_identity
                        and any(
                            goal == _goal_digest(target) and stalled >= RECOVERY_MODE_THRESHOLD
                            for goal, _, stalled in retained_recovery.goals
                        )
                    ):
                        stock_hedge = _StockHedge(
                            future, launched_at + _STOCK_HEDGE_DELAY, messages, route, exclude
                        )
                    cost = 2 if native or native_test or native_line or planner_input is not None or plan_private_method else 1
                    submitted += cost
                    launch_budget -= cost
            if not pending:
                for target in remaining:
                    if target in unavailable:
                        raise unavailable[target]
                break
            done, _ = wait(
                pending,
                timeout=min(budget.remaining(), 0.25),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            future = next(future for future in pending if future in done)
            if stock_hedge is not None and stock_hedge.primary is future:
                stock_hedge = None
            paired = future in hedge_pair
            hedge_pair.discard(future)
            (
                target,
                snapshot,
                continuation,
                named_definitions,
                advised,
                (rotation_key, rotated),
                localized_retry,
                line_retry,
                line_followup,
                allow_test_module,
                test_retry,
                raw_nested,
                repair,
            ) = pending.pop(future)
            test_raw_retry = test_retry if isinstance(test_retry, _RawTestRetry) else None
            test_line_snapshot = test_retry if isinstance(test_retry, LineRangeSnapshot) else None
            helper_authoring = test_retry if isinstance(test_retry, OperatorTestAssembly) else None
            scoped_line = line_retry is not None and line_retry.allowed_rows is not None
            if (
                paired or continuation is not None or advised
                or repair is not None or scoped_line or test_retry is not None
                or raw_nested is not None
            ) and snapshot is not current:
                continue
            if (
                snapshot is current and not scoped_line
                and (line_retry is not None or (localized_retry is not None and line_followup is None))
            ):
                interactive_target = target
                interactive_symbol = line_retry.snapshot.symbol if line_retry else localized_retry
            try:
                completion, native_candidate, fresh_advice, fresh_line_snapshot = (
                    future.result()
                )
            except ProviderExhausted as error:
                if paired and hedge_pair:
                    continue
                if target == _TEST_GOAL and rotated and snapshot is current:
                    # Try the configured routes again when the alternatives are exhausted.
                    test_model_failures = None
                    if helper_authoring is not None and helper_feedback:
                        # A runnable proposal can use its verification details even
                        # when all other configured models are unavailable.
                        helper_bad_model = ()
                elif (line_retry is not None and line_retry.direct_caller_only and snapshot is current
                      and line_retry.excluded_models and not line_retry.caller_routes_reopened):
                    # The scope changed. Once alternatives are exhausted, let a
                    # previously used free route see the caller-only prompt.
                    line_retries[target] = replace(line_retry, excluded_models=(), caller_routes_reopened=True)
                elif line_followup is not None:
                    line_retries[target] = line_followup
                elif line_retry is not None:
                    _LOG.info("Numbered review edit format had no available coding route")
                elif test_line_snapshot is not None:
                    _LOG.info("Numbered pytest edit format had no available coding route")
                elif test_raw_retry is not None:
                    _LOG.info("Raw pytest function edit had no available coding route")
                elif raw_nested is not None:
                    _LOG.info("Raw nested Python edit had no available coding route")
                elif rotated:
                    if snapshot is current:
                        # Reopen this goal's model choices for the rest of this draft.
                        # The next proposal uses the normal request budget and task exclusions.
                        model_failures[rotation_key] = None
                elif repair is None:
                    unavailable[target] = error
                continue
            except _RequestBudgetExceeded:
                continue
            if localized_retry_symbol(fresh_advice) is not None:
                pinned_restore_advice[target] = fresh_advice
            if completion is None:
                retry_advice[target] = fresh_advice
                _LOG.info("First numbered stock edit had no available coding route")
                continue
            active_line_snapshot = (
                line_retry.snapshot if line_retry else fresh_line_snapshot
            )
            fresh_stock = (
                snapshot is current
                and continuation is None
                and repair is None
                and failures.keys() == {target}
                and stock_fixtures.get((target, failures[target]))
                and completion.configured_model
            )
            if (
                fresh_stock
                and advised == "planning"
                and localized_retry_symbol(fresh_advice) is None
            ):
                unlocalized_plans.add(target)
                interactive_misses[target] += 1
            if native_candidate is not None:
                submitted -= 1  # The reserved text fallback was unused.
            unavailable.clear()
            if ready:
                continue
            if native_candidate is not None and snapshot is not current:
                continue
            routes = (
                line_routes if active_line_snapshot is not None and not scoped_line
                else last_routes
            )
            routes[target] = completion.route
            editor = RouteEvidence.from_completion(completion)
            if (
                test_raw_retry is not None
                and test_raw_retry.revision[1] != hashlib.sha256(test_draft.encode()).digest()
            ):
                continue
            if raw_nested is not None:
                raw_symbol, raw_revision, prior_raw = raw_nested
                raw_model = editor.route, editor.configured_model
                if not editor.configured_model or raw_model in prior_raw:
                    continue
            try:
                source = current if continuation is None else continuation[0]
                if repair is not None:
                    source = repair.sources
                if target == _TEST_GOAL:
                    source = {new_path: test_draft}
                try:
                    candidate = native_candidate
                    if raw_nested is not None:
                        candidate = apply_scoped_raw_definition(completion.content, source, raw_symbol)
                    if target == _TEST_GOAL:
                        if helper_authoring is not None:
                            if helper_bad_model == (editor.route, editor.configured_model):
                                raise ValueError("The fixture authoring retry selected an excluded coding model")
                            candidate = {new_path: helper_authoring.assemble(
                                completion.content, max_bytes=executor.limits.patch_bytes,
                            )}
                        elif test_raw_retry is not None:
                            pair = editor.route, editor.configured_model
                            if pair in test_raw_retry.excluded_models:
                                raise ValueError("Raw pytest edit requires a distinct coding route and model")
                            candidate = apply_scoped_raw_definition(
                                completion.content, source, test_raw_retry.snapshot.symbol, require_nested=False,
                            )
                        elif test_line_snapshot is not None:
                            candidate = apply_line_range_edit(
                                completion.content, source, test_line_snapshot,
                            )
                        elif candidate is None:
                            candidate = _apply_test_reply(
                                completion.content,
                                source,
                                new_path,
                                allow_module=(
                                    allow_test_module
                                    and original[new_path] == current[new_path] == ""
                                    and not _protected(new_path, task, new_test=True)
                                ),
                                failed_draft=bool(test_draft and test_diagnostic),
                                limit=executor.limits.patch_bytes,
                            )
                        if candidate == source:
                            raise ValueError("The pytest edit did not change the draft.")
                    if candidate is None and active_line_snapshot is not None:
                        if scoped_line and (line_retry.alternatives or line_retry.complete_single_method):
                            candidate = apply_offered_line_reply(
                                completion.content, source, (line_retry.snapshot, *line_retry.alternatives), block_pattern=_BLOCK,
                                full_method=line_retry.complete_single_method,
                            )
                        else:
                            candidate = apply_line_range_edit(
                                completion.content, source, active_line_snapshot,
                                allowed_rows=line_retry.allowed_rows if scoped_line else None,
                                full_method=scoped_line and line_retry.complete_single_method,
                            )
                    if candidate is None:
                        candidate = (
                            apply_definitions(
                                completion.content, source, only=localized_retry
                            )
                            if named_definitions
                            else apply_blocks(completion.content, source, new_path)
                        )
                except ValueError:
                    if repair is None and not scoped_line:
                        queue_iteration_retry(target, snapshot)
                    if target == _TEST_GOAL:
                        if snapshot is current:
                            test_format_failures += 1
                            count_test_model_failure(snapshot, editor, ("edit format",))
                            if helper_authoring is not None:
                                helper_setup_misses += 1
                                if editor.configured_model:
                                    helper_bad_model = editor.route, editor.configured_model
                            if test_raw_retry is not None and editor.configured_model:
                                pair = editor.route, editor.configured_model
                                prior = test_raw_retry.excluded_models
                                if pair not in prior:
                                    test_raw_attempts[test_raw_retry.revision] = (*prior, pair)
                        raw = completion.content or ""
                        _LOG.info(
                            "New pytest module reply via %s did not parse "
                            "(characters=%d fences=%d search_markers=%d "
                            "replace_markers=%d named_headers=%d diff_hunks=%d)",
                            completion.route, len(raw), len(_FENCE_LINE.findall(raw)),
                            raw.count("<<<<<<< SEARCH"), raw.count(">>>>>>> REPLACE"),
                            len(re.findall(r"(?m)^REPLACE ", raw)),
                            len(re.findall(r"(?m)^@@ ", raw)),
                        )
                    if stock_fixtures.get((target, failures.get(target))):
                        raw = completion.content or ""
                        _LOG.info(
                            "Rejected tracked-test edit format, characters=%d search_markers=%d "
                            "replace_markers=%d named_headers=%d diff_hunks=%d",
                            len(raw), raw.count("<<<<<<< SEARCH"), raw.count(">>>>>>> REPLACE"),
                            len(re.findall(r"(?m)^REPLACE ", raw)),
                            len(re.findall(r"(?m)^@@ ", raw)),
                        )
                    count_model_failure(rotation_key, snapshot, editor)
                    if fresh_advice is not None:
                        retry_advice[target] = fresh_advice
                    if (
                        not scoped_line
                        and target in failures
                        and check_label(commands[target[0]], ()) != "ruff"
                    ):
                        count_recovery(malformed, target)
                        if named_definitions and line_retry is None:
                            native_ready.add(target)
                    raise
                if target == _TEST_GOAL:
                    candidate = {**current, **candidate}
                    if (review.test_only or test_seed is not None) and any(
                        candidate[path] != current[path] for path in paths
                    ):
                        raise ValueError("Keep the verified production code unchanged and revise only the test module.")
                if (
                    failures
                    and needs_tests
                    and candidate[new_path] != current[new_path]
                ):
                    raise ValueError(
                        "Fix existing verification failures first, then add the new test module."
                    )
                proposed = patch_for(original, candidate)
                if len(proposed) > executor.limits.patch_bytes:
                    raise ValueError("Patch exceeds the configured byte limit")
                after_runs = {}
                static_runs = {}
                unused_import_checks = set()
                test_check_labels = set()
                focused_rejection = None
                after = {}
                remaining_commands = commands
                if (
                    target == _TEST_GOAL and focused_generated is not None
                    and all(candidate[path] == current[path] for path in paths)
                    and has_runnable_test(new_path, candidate[new_path].encode())
                ):
                    fast_runs, fast_checks = {}, []
                    after = review_evidence(
                        executor, task, archive, budget, proposed, (focused_generated,),
                        pytest_runs=fast_runs, test_check_labels=test_check_labels,
                        check_results=fast_checks,
                    )
                    fast_run = after_runs["focused"] = fast_runs.get(0)
                    if pinned_ancestor and _focused_test_body_assertion(
                        new_path, candidate[new_path], fast_run, after
                    ):
                        after_runs["focused"] = replace(fast_run, assertion_reached=True)
                    if fast_run and fast_run.named_cases is not None:
                        _LOG.info(
                            "Focused generated pytest cases: %d passed, %d failed",
                            fast_run.passed, len(fast_run.failed),
                        )
                    remaining_commands = tuple(
                        command for command in commands if command != focused_generated
                    )
                    if after:
                        error = _VerificationRejected(check_failure_summary(task, fast_checks))
                        error.checks = tuple(fast_checks)
                        try:
                            budget.reject(error, editor)
                        except AttemptRejected as rejection:
                            focused_rejection = rejection
                if not after:
                    after = review_evidence(
                        executor,
                        task,
                        archive,
                        budget,
                        proposed,
                        remaining_commands,
                        pytest_runs=after_runs,
                        static_runs=static_runs,
                        private_proposal=repair is not None and repair.stock_origin,
                        **({"test_check_labels": test_check_labels} if target == _TEST_GOAL else {}),
                        **(
                            {"unused_import_checks": unused_import_checks}
                            if repair is not None and repair.dependency_seed else {}
                        ),
                    )
            except _PrivateCheckOutputExceeded as error:
                budget.remaining()
                regression_repair = _retry_stock_repair(
                    repair,
                    target,
                    f"{error} Try a different fix against the unchanged stock-passing draft.",
                    named_definitions,
                )
                if regression_repair is not None and editor.configured_model:
                    model = (editor.route, editor.configured_model)
                    regression_repair = replace(
                        regression_repair,
                        excluded_models=tuple(
                            dict.fromkeys((*repair.excluded_models, model))
                        ),
                    )
                _LOG.info("Rejected private review proposal: %s", error)
                continue
            except ValueError as error:
                if raw_nested is not None:
                    raw_nested_attempts[raw_revision] = (*prior_raw, raw_model)
                if fresh_stock:
                    interactive_misses[target] += 1
                if fresh_line_snapshot is not None and snapshot is current:
                    retry_advice[target] = fresh_advice
                if line_followup is not None and snapshot is current:
                    line_retries[target] = line_followup
                notes[target] = str(error)[:400]
                retry_numbered_route(target, line_retry, editor, malformed=True, range_error=error)
                if repair is not None and repair.stock_origin:
                    regression_repair = _retry_stock_repair(
                        repair,
                        target,
                        notes[target]
                        + ". Use the unchanged stock-passing draft and follow the requested edit format.",
                        not named_definitions,
                    )
                elif repair is not None and repair.private_symbol:
                    _LOG.info("Private method repair outcome: malformed, attempt=%d", repair.retries + 1)
                    regression_repair = _retry_private_method(
                        repair, target, editor, "Previous proposal could not be applied. Use this tentative checkout.",
                    )
                _LOG.info(
                    "Review edit could not be applied for %s: %s",
                    target[1],
                    str(error).split(":", 1)[0][:120],
                )
                continue
            if target == _TEST_GOAL:
                next_draft = candidate[new_path]
                next_rejection = focused_rejection
                next_case_runs = after_runs
                next_ancestor_feedback = ""
                next_passed_ancestor = False
                result = None
                if after:
                    _LOG.info(
                        "New pytest module verification failed: %s",
                        "; ".join(sorted(test_check_labels)) or "check failure",
                    )
                    next_diagnostic = _failed_test_feedback(after, new_path)
                elif not has_runnable_test(new_path, next_draft.encode()):
                    next_diagnostic = (
                        "Add at least one runnable pytest test to this draft."
                    )
                else:
                    try:
                        result = _finish_review(
                            executor, task, archive, budget, proposed, editor, version
                        )
                    except AttemptRejected as error:
                        rejection = error.__cause__
                        if not isinstance(rejection, _VerificationRejected):
                            raise
                        next_rejection = error
                        if run := _focused_generated_run(rejection.checks, focused_generated):
                            next_case_runs["focused"] = run
                        failures_by_check = {}
                        for index, check in enumerate(rejection.checks):
                            if not check.returncode:
                                continue
                            if pinned_ancestor and check.command == ("verify-regression", task.regression_base):
                                next_ancestor_feedback = excerpt(check.output, 420)
                                next_passed_ancestor = "The new tests also pass on the ancestor." in check.output
                                test_check_labels.add(
                                    "ancestor/tests passed on both revisions"
                                    if next_passed_ancestor
                                    else "ancestor/no verified assertion failure"
                                )
                                continue
                            nodes = [node for _, node in _TEST_RESULT.findall(check.output)]
                            categories = set()
                            for node in nodes:
                                detail = _test_detail(check.output, node)
                                failures_by_check[index, node] = detail
                                if kind := _named_test_diagnostic_kind(detail):
                                    categories.add(kind)
                            if not nodes:
                                detail, kind = _generic_verifier_failure(
                                    check.command, check.returncode, check.output,
                                )
                                failures_by_check[index, f"check {index + 1}"] = detail if kind else check.output
                                if kind:
                                    categories.add(kind)
                            test_check_labels.add(_test_diagnostic_label(check.command, check.returncode, categories))
                        _LOG.info(
                            "Final verification rejected the new pytest module: %s",
                            "; ".join(sorted(test_check_labels)) or "check failure",
                        )
                        next_diagnostic = (
                            _failed_test_feedback(failures_by_check, new_path) + "\n" + str(rejection)
                        )
                    else:
                        next_case_runs["focused"] = _focused_generated_run(
                            result.checks, focused_generated
                        )
                candidate_run = next_case_runs.get("focused")
                smoke = lone_both_pass_case() if snapshot is current else None
                if helper_authoring is not None and result is None:
                    outcome = "Current verification rejected the function. Correct its setup or assertions."
                    if next_passed_ancestor:
                        outcome = "The function passed on both revisions. Strengthen its scenario and assertions."
                    elif next_ancestor_feedback:
                        outcome = "The pinned run did not establish an assertion failure in the test."
                    # Put pytest's rewritten assertion or exception before traceback
                    # frames so it survives excerpting on long execution diagnostics.
                    reasons = [
                        line.strip() for (_, node), detail in after.items()
                        if node.removeprefix("./").startswith(new_path + "::")
                        for line in detail.splitlines()
                        if re.match(r"^E\s+(?:assert\b|[\w.]*(?:Error|Exception):)", line)
                    ]
                    if next_passed_ancestor or (candidate_run and candidate_run.assertion_reached):
                        # Count verifier observations, independent of test names and
                        # proposal formatting. Repeated both-green tests are one outcome.
                        signature = ("both passed",) if next_passed_ancestor else tuple(
                            re.sub(r"0x[0-9a-fA-F]{6,}", "<address>", item) for item in reasons[:2]
                        ) or ("assertion reached",)
                        helper_outcomes[signature] += 1
                        helper_setup_misses = 0
                    else:
                        helper_setup_misses += 1
                    detail = next_ancestor_feedback or next_diagnostic
                    helper_feedback = outcome + "\n" + excerpt("\n".join((*reasons[:2], detail)), 750)
                    helper_function = completion.content or ""
                    if editor.configured_model:
                        helper_bad_model = editor.route, editor.configured_model
                if helper_authoring is not None and (
                    not (candidate_run and candidate_run.passed) or (smoke and result is None)
                ):
                    notes[target] = (
                        "The fixture-based proposal did not pass regression verification. "
                        "Continue from the retained unpublished draft.\n" + excerpt(next_diagnostic, 550)
                    )
                    continue
                lost_test = (
                    _lost_passing_test(test_case_runs, next_case_runs, new_path)
                    if allow_test_module and test_checkout_snapshot is current else None
                )
                retire_smoke = bool(
                    smoke and lost_test == smoke and result is not None
                    and candidate_run and candidate_run.named_cases and candidate_run.passed
                    and not candidate_run.failed and not candidate_run.deselected
                    and any(
                        check.command == ("verify-regression", task.regression_base) and check.returncode == 0
                        for check in result.checks
                    )
                )
                previous_run = test_case_runs.get("focused")
                if (
                    test_raw_retry is not None and previous_run and previous_run.passed
                    and (not candidate_run or candidate_run.named_cases is None or candidate_run.deselected)
                ):
                    notes[target] = (
                        "The scoped edit could not verify the already-passing cases. The previous draft was retained.\n"
                        + excerpt(next_diagnostic, 550)
                    )
                    continue
                if prior_test_fallback is not None and snapshot is current:
                    if candidate_run and candidate_run.passed and not lost_test:
                        prior_test_fallback = None
                    elif focused_rejection is not None and not (candidate_run and candidate_run.assertion_reached):
                        (
                            test_draft, test_diagnostic, test_ancestor_feedback,
                            test_rejection, test_case_runs, test_passed_ancestor,
                        ) = prior_test_fallback
                        prior_test_fallback = None
                        notes[target] = (
                            "The seeded edit failed before reaching a test-body assertion. "
                            "Restored the previous unpublished draft."
                        )
                        _LOG.info("Restored the previous pytest draft after the seeded edit regressed setup")
                        continue
                if (
                    production_regression and allow_test_module and snapshot is current
                    and focused_rejection is not None
                    and not (candidate_run and (candidate_run.assertion_reached or candidate_run.passed))
                    and any(
                        kind in _PRIOR_SEED_SETUP_ERRORS
                        for label in test_check_labels if label.startswith("pytest/")
                        for kind in label.partition(": ")[2].split(", ")
                    )
                ):
                    test_setup_failures += 1
                elif candidate_run and (candidate_run.assertion_reached or candidate_run.passed):
                    test_setup_failures = 0
                if snapshot is current:
                    previous_passed = previous_run.passed if previous_run else 0
                    improved = candidate_run and (
                        candidate_run.passed > previous_passed
                        or (
                            test_draft and candidate_run.passed == previous_passed
                            and candidate_run.assertion_reached
                            and not (previous_run and previous_run.assertion_reached)
                        )
                    )
                    if improved and not lost_test:
                        test_model_failures = Counter()
                    elif result is None:
                        count_test_model_failure(snapshot, editor, test_check_labels)
                if (
                    pinned_ancestor and allow_test_module and test_checkout_snapshot is current
                    and previous_run and previous_run.assertion_reached and focused_rejection is not None
                    and not (candidate_run and (candidate_run.passed or candidate_run.assertion_reached))
                    and not _focused_output_reports_pass(fast_checks[0].output, new_path)
                ):
                    notes[target] = (
                        "The last edit regressed before reaching a test-body assertion. "
                        "Revise the retained draft using its verified assertion failure.\n"
                        + excerpt(next_diagnostic, 550)
                    )
                    _LOG.info("Retained the unpublished pytest draft that reached a test-body assertion")
                    continue
                if (
                    test_ancestor_feedback and previous_run and previous_run.named_cases
                    and not previous_run.failed and focused_rejection is not None
                    and (not candidate_run or candidate_run.named_cases is None)
                ):
                    notes[target] = (
                        "The last edit failed before completing the focused pytest run. "
                        "Revise the retained runnable draft using this latest feedback:\n"
                        + excerpt(next_diagnostic, 550)
                    )
                    _LOG.info("Retained the runnable pytest draft awaiting pinned ancestor proof")
                    continue
                if lost_test and not retire_smoke:
                    notes[target] = (
                        f"The last edit regressed or removed the already-passing case {lost_test}. "
                        "This prompt shows the retained draft and its original failure. Preserve the passing case."
                    )
                    _LOG.info("Retained the prior unpublished draft after losing a verified passing pytest case")
                    continue
                if result is not None:
                    if next_draft == prior_test_source:
                        notes[target] = (
                            "Strengthen the predecessor seed with an assertion for the reviewed behavior."
                        )
                        continue
                    if test_seed is not None and (
                        next_draft == test_seed.source
                        or not _covers_generated_seed(
                            next_case_runs.get("focused"), new_path,
                            test_seed.passing + test_seed.omitted,
                        )
                    ):
                        notes[target] = (
                            "Expand the test module and preserve these verified passing and "
                            "omitted cases: " + ", ".join(test_seed.passing + test_seed.omitted)
                        )
                        continue
                    coverage_revision = None
                    if fixture_requests:
                        test_syntax = ast.dump(ast.parse(next_draft))
                        coverage_revision = test_checkout_digest, hashlib.sha256(test_syntax.encode()).digest()
                    missing = known_fixture_omissions.get(coverage_revision, ())
                    if (
                        fixture_requests
                        and not missing
                        and (known_fixture_omissions or submitted < max_proposals)
                        and budget.model_requests_remaining() >= (1 if known_fixture_omissions else 3)
                    ):
                        missing = audit_fixture_requests(
                            executor, budget, fixture_requests, new_path, next_draft,
                            next_case_runs.get("focused"),
                        )
                        route_trace, usage = budget.route_usage()
                        result = replace(result, route_trace=route_trace, usage=usage)
                    if not missing:
                        return result
                    known_fixture_omissions[coverage_revision] = missing
                    next_diagnostic = (
                        "Independent pytest verification passed. Preserve the passing cases. "
                        "Add coverage for these explicit review fixture requests:\n"
                        + "\n".join(f"- {request}" for request in missing)
                    )
                    if retire_smoke:
                        notes[target] = next_diagnostic
                        continue
                    _LOG.info("Retained passing pytest draft with %d missing fixture requests", len(missing))
                if allow_test_module and snapshot is current:
                    retain_test_seed(candidate, next_case_runs)
                    if production_regression and len(review.feedback) == 1 and focused_rejection is not None:
                        next_ancestor_feedback = failed_assertion_hint(next_draft, candidate_run, after)
                test_draft = next_draft
                test_rejection = next_rejection
                test_diagnostic = next_diagnostic
                test_ancestor_feedback = next_ancestor_feedback
                test_passed_ancestor = snapshot is current and next_passed_ancestor
                test_case_runs = next_case_runs if allow_test_module else {}
                notes.pop(target, None)
                continue
            # Never trade a passing operator test for another one. Stock diagnostics
            # can change while a behavior is corrected; finish those before publishing.
            old_operator = {
                item for item in failures if item[0] < len(task.pull_checks)
            }
            new_operator = {item for item in after if item[0] < len(task.pull_checks)}
            introduced = new_operator - old_operator
            gained = old_operator - new_operator
            if stock_fixtures.get((target, failures.get(target))):
                # Record checked behavior, without exposing a model reply or test values.
                run = after_runs.get(target[0])
                check_types = sorted({check_label(commands[number], ()) or "configured" for number, _ in after})
                _LOG.info(
                    "Tracked repair verification via %s: repository pytest passed=%s failed=%s; "
                    "failing review cases=%d; outstanding checks=%s",
                    completion.route, run.passed if run else "unknown",
                    len(run.failed) if run else "unknown", len(new_operator), ", ".join(check_types) or "none",
                )
            elif target in old_operator and len(old_operator) == 1:
                remaining_checks = {check_label(commands[number], ()) or "configured" for number, _ in after}
                _LOG.info(
                    "Final review case verification via %s: selected case %s; remaining review cases=%d; checks=%s",
                    completion.route, "failed" if target in after else "absent from failure list",
                    len(new_operator), ", ".join(sorted(remaining_checks)) or "none",
                )
            static_failures = _localized_static_failures(
                after.keys() - new_operator, static_runs, candidate, paths
            )
            stock_gain = (
                not old_operator
                and snapshot is current
                and qualified_stock_gain(target, after, after_runs, introduced, static_failures)
            )
            if repair is not None:
                improved = (
                    not (repair.gained & new_operator)
                    and introduced <= repair.introduced
                    and _named_operator_cases(old_operator | new_operator, commands)
                    and (
                        introduced < repair.introduced
                        or (
                            repair.attempts == 0
                            and assertion_progress(
                                assertions,
                                task.pull_checks,
                                target,
                                repair.failures,
                                after,
                            )
                        )
                    )
                )
                if repair.stock_origin:
                    pending_static = _localized_static_failures(
                        repair.static_failures.keys() & after.keys(),
                        static_runs,
                        candidate,
                        paths,
                    )
                    verified = (
                        _preserved_stock_gain(
                            repair.origin, pytest_runs, after_runs, after
                        )
                        and _verified_operator_runs(
                            pytest_runs,
                            after_runs,
                            new_operator,
                            task.pull_checks,
                            operator_cases,
                        )
                        and (after.keys() - new_operator)
                        <= (repair.failures.keys() - repair.introduced)
                        and introduced <= repair.introduced
                        and pending_static is not None
                        and all(
                            amount <= repair.static_failures[key]
                            for key, amount in pending_static.items()
                        )
                    )
                    improved = verified and (
                        introduced < repair.introduced
                        or sum(pending_static.values())
                        < sum(repair.static_failures.values())
                    )
                    if improved and repair.stock_collection_id is not None:
                        try:
                            improved = (
                                editor.role == "coding"
                                and bool(editor.configured_model)
                                and identify_stock_collection(
                                    repair.origin, proposed, _GuardBudget(budget, reserve=25)
                                ) == repair.stock_collection_id
                            )
                        except Cancelled:
                            raise
                        except (ExecutionError, ValueError):
                            budget.remaining()
                            improved = False
                    if not improved:
                        reason = (
                            "The selected case still fails on the proposed edit."
                            if verified and target in after
                            else "The edit regressed a passing check or could not be verified."
                        )
                        stock_failed = sum(
                            number >= len(task.pull_checks)
                            and check_label(commands[number], ()) == "pytest"
                            for number, _ in after
                        )
                        static_failed = {
                            number for number, _ in after
                            if check_label(commands[number], ()) in {"ruff", "mypy"}
                        }
                        _LOG.info(
                            "Private stock repair rejected for %s (%d stock failures, %d review failures, %d static checks)",
                            excerpt(target[1].partition("[")[0].rsplit("/", 1)[-1], 160),
                            stock_failed,
                            sum(number not in static_failed for number, _ in new_operator),
                            len(static_failed),
                        )
                        regression_repair = _retry_stock_repair(
                            repair,
                            target,
                            reason
                            + " Try a different fix against the unchanged stock-passing draft."
                            + private_retry_feedback(repair, target, after, after_runs, static_runs),
                            named_definitions,
                            checked=True,
                        )
                        if regression_repair is not None:
                            continue
                    else:
                        _LOG.info(
                            "Private stock repair progressed: %d review case(s) and %d static check(s) remain",
                            len(introduced),
                            len(pending_static),
                        )
                else:
                    pending_static = {}
                if repair.private_symbol:
                    gained_status = _private_case_outcome(repair.origin, after, after_runs, commands)
                    control_status = _private_case_outcome(target, after, after_runs, commands)
                    _LOG.info("Private method repair outcome: executable, attempt=%d, target=%s, control=%s, failing_checks=%d",
                              repair.retries + 1, gained_status, control_status, len(after))
                if not improved and repair.private_symbol:
                    observed = after.get(target, "").partition("\nObserved pytest failure: ")[2]
                    reason = f"Last discarded edit: original target {gained_status}; previously passing control {control_status}."
                    if observed:
                        reason += " Observed pytest failure: " + excerpt(observed, 300)
                    regression_repair = _retry_private_method(repair, target, editor, reason, checked=True)
                    if regression_repair:
                        continue
                if not improved or introduced or pending_static:
                    if improved and (repair.stock_origin or repair.attempts == 0):
                        regression_repair = replace(
                            repair,
                            sources=candidate,
                            failures=after,
                            gained=gained,
                            introduced=introduced,
                            attempts=repair.attempts + 1,
                            retries=0,
                            retry_note="",
                            retry_named=False,
                            excluded_models=(),
                            retry_target=None,
                            stalled_edits=0,
                            static_failures=pending_static,
                            unused_import_checks=frozenset(unused_import_checks),
                        )
                        if snapshot is current and source is repair.sources:
                            save_private_progress(repair, regression_repair, proposed, editor)
                    else:
                        rejected[repair.origin] = _rejected_attempt(
                            current, candidate, repair.origin, failures, after
                        )
                    continue
            elif (
                (stock_gain or (introduced and gained))
                and snapshot is current
                and target not in repair_attempted
                and regression_repair is None
                and not any(
                    request[1] is current and request[-1] is not None
                    for request in pending.values()
                )
                and _named_operator_cases(old_operator | new_operator, commands)
            ):
                reconciled = None
                fresh_provider_edit = source is current and bool(
                    editor.configured_model
                )
                if stock_gain and (not static_failures or fresh_provider_edit):
                    static_locations = {
                        (path.removeprefix("/work/").removeprefix("./"), row)
                        for number, _ in static_failures or ()
                        for path, row in static_runs[number]
                    }
                    reconciled = reconcile_stock_gain(
                        target,
                        candidate,
                        {key: after[key] for key in introduced}
                        if fresh_provider_edit
                        else None,
                        static_locations=static_locations,
                    )
                if reconciled is None:
                    repair_attempted.add(target)
                    retry_advice.clear()
                    line_retries.clear()
                    continuations.pop(target, None)
                    admitted_static = (
                        static_failures if stock_gain and static_failures else {}
                    )
                    regression_repair = _RegressionRepair(
                        target,
                        candidate,
                        after,
                        gained,
                        introduced,
                        stock_origin=bool(stock_gain),
                        private_symbol=(
                            _private_changed_method(current, candidate)
                            if production_regression and not stock_gain and len(introduced) == 1 else None
                        ),
                        static_failures=admitted_static,
                    )
                    if (regression_repair.private_symbol and task.save_solver_recovery
                            and executor.runner is not executor.verification_runner
                            and 0 < len(proposed) <= PRIVATE_PATCH_BYTES and current is resumed_operator_checkout):
                        try:
                            cases = checked_tradeoff(proposed, after, after_runs, _GuardBudget(budget, wall_seconds=45))
                        except Cancelled:
                            raise
                        except (ExecutionError, OSError, ValueError):
                            budget.remaining()
                            cases = None
                        if cases:
                            tradeoff = PrivateCaseTradeoff(tradeoff_identity(), *map(_goal_digest, cases), proposed)
                            retain_recovery(durable=True)
                            _LOG.info("Saved an independently checked private one-for-one tradeoff")
                    if stock_gain:
                        _LOG.info(
                            "Entering private stock repair with %d review case(s) and %d static check(s)",
                            len(introduced),
                            len(admitted_static),
                        )
                    continue
                candidate, proposed, after, after_runs = reconciled
                new_operator, introduced = set(), set()
            if (
                fresh_stock
                and editor.configured_model
                and target in after
                and (introduced or static_runs)
            ):
                isolated = isolate_stock_hunk(
                    target, candidate, after, after_runs, introduced, static_runs
                )
                if isolated is not None:
                    (candidate, proposed, after, after_runs), _ = isolated
                    new_operator, introduced = set(), set()
                    _LOG.info("Full verification confirmed a gain from one stock repair hunk")
            progressed = new_operator < old_operator or (
                new_operator == old_operator and after.keys() < failures.keys()
            )
            partial = not progressed and assertion_progress(
                assertions, task.pull_checks, target, failures, after
            )
            stock_progress = ""
            isolated_stock_failure = (
                snapshot is current
                and is_isolated_stock_failure(target, after, after_runs)
            )
            if isolated_stock_failure:
                previous_site = stock_failure_site(target[1], failures[target])
                next_site = stock_failure_site(target[1], after[target])
                if previous_site and next_site:
                    _LOG.info(
                        "Tracked pytest verification for %s: %s -> %s",
                        target[1],
                        previous_site,
                        next_site,
                    )
                stock_progress = tracked_assertion_note(target, candidate, after)
                partial = bool(stock_progress)
                if partial:
                    _LOG.info("%s", stock_progress)
                elif source is current and repair is None and editor.configured_model:
                    composed = compose_stock_gain(target, candidate)
                    if composed is not None:
                        candidate, proposed, after, after_runs = composed
                        progressed = True
                        _LOG.info("Combined independently verified edits for %s", target[1])
            prefix_progress = ""
            if (
                not introduced
                and not (progressed or partial)
                and snapshot is current
                and target in old_operator
                and no_progress.get(target)
            ):
                prefix_progress = observe_prefix_progress(
                    executor,
                    task,
                    archive,
                    budget,
                    assertions,
                    fixtures,
                    target,
                    failures,
                    after,
                    patch,
                    proposed,
                )
            stale_regression = (
                snapshot is not current and not after.keys() <= failures.keys()
            )
            if (
                introduced
                or stale_regression
                or not (progressed or partial or prefix_progress)
            ):
                if raw_nested is not None:
                    if after.keys() == failures.keys():
                        raw_nested_attempts[raw_revision] = (*prior_raw, raw_model)
                    notes[target] = "The scoped helper replacement did not improve the verified checks."
                    continue
                count_model_failure(rotation_key, snapshot, editor)
                if after.keys() == failures.keys() or (line_retry and (line_retry.checkpoint_context or line_retry.direct_caller_only)):
                    retry_numbered_route(target, line_retry, editor, after=after, checked_candidate=candidate)
                if fresh_stock:
                    interactive_misses[target] += 1
                if snapshot is current and (
                    (target in old_operator and "::" in target[1])
                    or (
                        target in after
                        and stock_fixtures.get((target, failures.get(target)))
                    )
                ):
                    count_recovery(no_progress, target)
                if (
                    fresh_advice is not None
                    and target in after
                    and (not introduced or localized_retry_symbol(fresh_advice) is not None)
                ):
                    retry_advice[target] = fresh_advice
                if line_followup is not None and target in after:
                    line_retries[target] = line_followup
                if (
                    repair is None and not scoped_line and after.keys() == failures.keys()
                    and queue_iteration_retry(target, snapshot)
                ):
                    continue
                if (
                    continuation is None
                    and not scoped_line
                    and snapshot is current
                    and (target in old_operator or (isolated_stock_failure and mock_diagnostics.get(target)))
                    and "::" in target[1]
                    and after.keys() == failures.keys()
                ):
                    continuations.setdefault(target, (candidate, after[target]))
                    if fresh_advice is None:
                        continue
                # Keep only bounded prompt evidence, never the candidate or its full patch.
                rejected[target] = _rejected_attempt(
                    source, candidate, target, failures, after
                )
                notes.pop(target, None)
                _LOG.info(
                    "Review goal %s: no verified improvement via %s",
                    target[1],
                    completion.route,
                )
                continue
            if paired and hedge_pair:
                sibling = hedge_pair.pop()
                pending.pop(sibling)
                if not sibling.cancel() and not sibling.done():
                    retired = sibling
            current, failures, patch, actor = candidate, after, proposed, editor
            pytest_runs = after_runs
            start_new_draft(target)
            if prefix_progress:
                notes[target] = prefix_progress
            if not failures and (
                not task.require_new_tests
                or has_runnable_test(new_path, current[new_path].encode())
            ):
                # The final verifier includes an isolated run of new tests. Keep the
                # last partial checkpoint until that independent gate succeeds.
                continue
            _LOG.info(
                "Checkpointing review progress from %s: %d failed checks/tests remain",
                completion.route,
                len(after),
            )
            progress_note = (
                prefix_progress
                or stock_progress
                or (
                    "An operator case reached a later assertion and still failed."
                    if partial
                    else ""
                )
            )
            save_verified_progress(patch, progress_note, editor)
    finally:
        # A running gateway request is already bounded by the shared deadline.
        pool.shutdown(wait=False, cancel_futures=True)
        retain_recovery()
    if test_rejection is not None:
        raise test_rejection
    if patch and (submitted >= max_proposals or not budget.model_requests_remaining()):
        budget.exhaust_review_round(editor)
    budget.reject(
        _CandidateOutputError(
            "Review solver could not complete a repair; outstanding checks still fail"
        ),
        editor,
    )
