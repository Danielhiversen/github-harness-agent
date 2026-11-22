"""Observe one selected pytest mapping counter without serializing mappings."""

import ast
import re
from itertools import islice
from pathlib import Path
from types import GetSetDescriptorType

MAX_EVENTS = 8
MAX_STACK = 3
MAX_PATH = 120
_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", re.ASCII)
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
_SENSITIVE = re.compile(
    r"secret|token|passwd|password|credential|api.?key|authorization|cookie|email|phone|identity|account|"
    r"customer|user|tenant|session|uuid|(?:^|_)id(?:$|_)",
    re.I,
)
_COMPARISON = re.compile(r"assert (-?[0-9]{1,8}) == (-?[0-9]{1,8})")


def count(value):
    return type(value) is int and 0 <= value <= 10_000_000


def _assertion(path, node):
    try:
        with path.open("rb") as source:
            raw = source.read(128 * 1024 + 1)
        if len(raw) > 128 * 1024:
            return None
        scope = ast.parse(raw)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    qualified = node.partition("[")[0].split("::")[1:]
    for name in qualified:
        matches = [item for item in scope.body if getattr(item, "name", None) == name]
        if len(matches) != 1 or not isinstance(
            matches[0], (ast.ClassDef, ast.FunctionDef)
        ):
            return None
        scope = matches[0]
    if not isinstance(scope, ast.FunctionDef):
        return None
    pending = list(scope.body)
    matches = []
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not isinstance(item, ast.Assert):
            pending.extend(ast.iter_child_nodes(item))
            continue
        match item.test:
            case ast.Compare(left=left, ops=[ast.Eq()], comparators=[right]):
                for index, (subscript, target) in enumerate(
                    ((left, right), (right, left))
                ):
                    if (
                        isinstance(subscript, ast.Subscript)
                        and isinstance(subscript.slice, ast.Constant)
                        and type(key := subscript.slice.value) is str
                        and _KEY.fullmatch(key)
                        and not _SENSITIVE.search(key)
                        and isinstance(target, ast.Constant)
                        and count(target.value)
                    ):
                        matches.append(
                            (item.lineno, item.end_lineno, key, target.value, 1 - index)
                        )
    return matches[0] if len(matches) == 1 else None


def _instance_values(owner):
    # Bypass application attribute lookup, including a custom __dict__ property.
    cls = type(owner)
    for base in type.__getattribute__(cls, "__mro__"):
        descriptor = type.__getattribute__(base, "__dict__").get("__dict__")
        if descriptor is not None:
            if type(descriptor) is GetSetDescriptorType:
                fields = descriptor.__get__(owner)
                if type(fields) is dict:
                    return islice(dict.values(fields), 32)
            break
    return ()


def _mapping_count(mapping, key):
    for field, value in dict.items(mapping):
        if type(field) is str and field == key:
            return value if count(value) else NotImplemented
    return None


class CounterTrace:
    def __init__(self, path, node, reviewed_paths):
        self.path = path
        self.assertion = _assertion(path, node)
        self.paths = {}
        for name in reviewed_paths:
            self.paths[name] = self.paths[str(Path(name).resolve())] = name
        self.events = []
        self.mappings = {}
        self.changed = None
        self.actual = None
        self.seen_assert = self.visited = self.overflow = False
        self.samples = 0

    def _locations(self, frame, row):
        locations = []
        for _ in range(12):
            if frame is None:
                break
            path = self.paths.get(frame.f_code.co_filename)
            name = frame.f_code.co_qualname.replace(".<locals>.", ".")
            if path and len(name) <= 96 and _NAME.fullmatch(name):
                locations.append([path, name, row])
                if len(locations) == MAX_STACK:
                    break
            frame = frame.f_back
            if frame is not None:
                row = frame.f_lineno
        return locations

    def _sample(self, frame, row):
        try:
            self._record(frame, row)
        except (AttributeError, TypeError, RuntimeError):
            # This observation must not change test execution if a frame is not
            # inspectable or a concurrent mutation invalidates dict iteration.
            self.overflow = True

    def _record(self, frame, row):
        self.samples += 1
        if self.samples > 5000:
            self.overflow = True
            return
        self.visited = True
        local = frame.f_locals
        mappings = list(islice(local.values(), 32))
        if "self" in local:
            mappings.extend(_instance_values(local["self"]))
        for mapping in mappings:
            if type(mapping) is not dict or len(mapping) > 128:
                continue
            identity = id(mapping)
            if identity not in self.mappings and len(self.mappings) >= 64:
                continue
            value = _mapping_count(mapping, self.assertion[2])
            previous = self.mappings.get(identity)
            self.mappings[identity] = (mapping, value)
            if (
                previous is None
                or value == previous[1]
                or NotImplemented in (value, previous[1])
            ):
                continue
            locations = self._locations(frame, row)
            if locations:
                if len(self.events) == MAX_EVENTS:
                    self.overflow = True
                    return
                if self.changed is not None and self.changed != identity:
                    self.overflow = True
                    return
                self.changed = identity
                self.events.append([locations, previous[1], value])

    def tracer(self, code):
        if self.assertion is None:
            return None

        def dispatch(frame, event, _value):
            if self.overflow or event != "call":
                return None
            if frame.f_code is code:

                def test_location(frame, event, _value):
                    if event == "line" and frame.f_lineno == self.assertion[0]:
                        self.seen_assert = True
                    return test_location

                return test_location
            if frame.f_code.co_filename not in self.paths:
                return None
            previous = frame.f_lineno
            self._sample(frame, previous)

            def observe(frame, event, _value):
                nonlocal previous
                if self.overflow:
                    return None
                if event in {"line", "return"}:
                    self._sample(frame, previous)
                    previous = frame.f_lineno
                return observe

            return observe

        return dispatch

    def failure(self, report, call):
        if (
            self.assertion is None
            or call.excinfo is None
            or call.excinfo.type is not AssertionError
        ):
            return
        start, end, _, expected, position = self.assertion
        crash = getattr(report.longrepr, "reprcrash", None)
        if (
            crash is None
            or not start <= crash.lineno <= end
            or Path(crash.path).resolve() != self.path
        ):
            return
        first_line = (
            crash.message[:200].partition("\n")[0].removeprefix("AssertionError: ")
        )
        if match := _COMPARISON.fullmatch(first_line):
            values = tuple(map(int, match.groups()))
            actual = values[1 - position]
            if values[position] == expected and count(actual) and actual != expected:
                self.actual = actual

    def report(self):
        if (
            self.assertion is None
            or not self.seen_assert
            or not self.visited
            or self.overflow
        ):
            return None
        target = self.assertion[3] if self.actual is None else self.actual
        if self.changed is not None:
            aligned = self.mappings[self.changed][1] == target
        else:
            aligned = any(value == target for _, value in self.mappings.values())
        if not aligned:
            return None
        return [self.assertion[0], self.actual is not None, self.events]


def valid_report(data, paths):
    if type(data) is not list or len(data) != 3:
        return False
    line, mismatch, events = data
    if (
        type(line) is not int
        or not 1 <= line <= 128 * 1024
        or type(mismatch) is not bool
    ):
        return False
    if type(events) is not list or len(events) > MAX_EVENTS:
        return False
    for event in events:
        if type(event) is not list or len(event) != 3:
            return False
        stack, before, after = event
        if (
            type(stack) is not list
            or not 1 <= len(stack) <= MAX_STACK
            or any(value is not None and not count(value) for value in (before, after))
            or before == after
        ):
            return False
        for location in stack:
            if type(location) is not list or len(location) != 3:
                return False
            path, function, row = location
            if (
                type(path) is not str
                or path not in paths
                or type(function) is not str
                or len(function) > 96
                or not _NAME.fullmatch(function)
                or type(row) is not int
                or not 1 <= row <= 256 * 1024
            ):
                return False
    return True
