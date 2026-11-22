"""Read which clauses a plain Python assertion actually evaluated in an auxiliary replay."""

import argparse
import ast
import dis
import json
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrefixReadout:
    line: int
    clauses: tuple[str, ...]
    values: tuple[bool | None, ...]

    def passed_prefix(self):
        """Return a comparable partial score only for an observed false clause."""
        if False not in self.values:
            return None
        failed = self.values.index(False)
        if all(value is True for value in self.values[:failed]) and all(
            value is None for value in self.values[failed + 1 :]
        ):
            return failed
        return None


def prefix_improved(before, after):
    """Compare readouts for the same already matched test selector and assertion."""
    if (
        before.line != after.line
        or before.clauses != after.clauses
        or len(before.values) != len(after.values)
    ):
        return False
    previous, current = before.passed_prefix(), after.passed_prefix()
    return previous is not None and current is not None and current > previous


def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _clauses(source, statement):
    return tuple(
        (ast.get_source_segment(source, term) or "")[:240]
        for term in statement.test.values
    )


def _assertion(source, test_name):
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == test_name
    ]
    if len(functions) != 1:
        return None
    assertions = [
        item
        for item in functions[0].body
        if isinstance(item, ast.Assert)
        and isinstance(item.test, ast.BoolOp)
        and isinstance(item.test.op, ast.And)
        and 2 <= len(item.test.values) <= 8
    ]
    if len(assertions) != 1:
        return None
    statement = assertions[0]
    for value in statement.test.values:
        if any(
            isinstance(node, (ast.BoolOp, ast.IfExp, ast.Lambda))
            or isinstance(node, ast.Compare)
            and len(node.ops) != 1
            for node in ast.walk(value)
        ):
            return None
    return statement


def _branches(code, terms):
    instructions = list(dis.get_instructions(code))
    by_offset = {item.offset: item for item in instructions}
    positions = {_span(term): index for index, term in enumerate(terms)}
    branches = {}
    for offset, instruction in enumerate(instructions[:-1]):
        if instruction.opname not in {
            "POP_JUMP_IF_FALSE",
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_FORWARD_IF_FALSE",
            "POP_JUMP_FORWARD_IF_TRUE",
        }:
            continue
        index = positions.get(tuple(instruction.positions))
        if index is None:
            continue
        if index in branches or not isinstance(instruction.argval, int):
            return {}
        next_offset = instructions[offset + 1].offset
        if instruction.opname.endswith("IF_FALSE"):
            passed, failed = next_offset, instruction.argval
        else:
            passed, failed = instruction.argval, next_offset
        branches[index] = (instruction.offset, passed, failed)

    if len(branches) != len(terms):
        return {}
    ordered = [branches[index] for index in range(len(terms))]
    failure = by_offset.get(ordered[0][2])
    if failure is None or failure.opname != "LOAD_ASSERTION_ERROR":
        return {}
    if any(
        failed != failure.offset
        or not jump < passed
        or (index < len(ordered) - 1 and not passed <= ordered[index + 1][0])
        for index, (jump, passed, failed) in enumerate(ordered)
    ):
        return {}
    return {
        jump: (index, passed, failed)
        for index, (jump, passed, failed) in enumerate(ordered)
    }


class _AssertionTrace:
    def __init__(self, path, test_name):
        self.path = Path(path).resolve()
        self.test_name = test_name
        self.results = []
        self.previous = None
        self.enabled = False
        self.statement = None
        try:
            info = self.path.stat()
            if stat.S_ISREG(info.st_mode) and info.st_size <= 128 * 1024:
                from tokenize import open as open_python

                with open_python(self.path) as file:
                    self.source = file.read(128 * 1024 + 1)
                self.statement = _assertion(self.source, test_name)
        except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError):
            pass

    def __enter__(self):
        self.previous = sys.gettrace()
        if self.statement is not None and self.previous is None:
            self.enabled = True
            # CPython 3.12 enables opcode notifications at trace installation
            # only when an active frame already asks for them.
            frame = sys._getframe()
            frame.f_trace_opcodes = True
            try:
                sys.settrace(self._call)
            finally:
                frame.f_trace_opcodes = False
        return self

    @property
    def eligible(self):
        return self.statement is not None

    def __exit__(self, *_):
        if self.enabled:
            sys.settrace(self.previous)
            self.enabled = False

    def _call(self, frame, event, _):
        if (
            event != "call"
            or frame.f_code.co_name != self.test_name
            or Path(frame.f_code.co_filename).resolve() != self.path
        ):
            return None
        terms = self.statement.test.values
        branches = _branches(frame.f_code, terms)
        if not branches:
            return None
        frame.f_trace_opcodes = True
        frame.f_trace_lines = False
        values = [None] * len(terms)
        pending = None

        def trace(frame, event, _):
            nonlocal pending
            if event == "exception":
                pending = None
            elif event == "opcode":
                offset = frame.f_lasti
                if pending is not None:
                    index, passed, failed = pending
                    pending = None
                    if offset not in {passed, failed}:
                        return None
                    values[index] = offset == passed
                    if not values[index] or index == len(terms) - 1:
                        clauses = _clauses(self.source, self.statement)
                        self.results.append(
                            PrefixReadout(self.statement.lineno, clauses, tuple(values))
                        )
                        return None
                pending = branches.get(offset)
            return trace

        return trace


def trace_compound_assert(path, test_name):
    """Observe a direct `assert A and B…` without editing or evaluating its source."""
    return _AssertionTrace(path, test_name)


def main():
    """Run one trusted test selector in a disposable verification workspace."""
    import pytest

    sys.path.insert(0, str(Path.cwd()))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("test_name")
    options = parser.parse_args()
    if not options.fixture.is_absolute() or not options.test_name.isidentifier():
        parser.error("Use an absolute operator fixture path and one test function name")

    readout = trace_compound_assert(options.fixture, options.test_name)
    if not readout.eligible:
        return int(pytest.ExitCode.NO_TESTS_COLLECTED)

    class SingleTest:
        def pytest_collection_modifyitems(self, items):
            if len(items) != 1 or items[0].name != options.test_name:
                items.clear()

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_call(self):
            with readout:
                return (yield)

    status = pytest.main(
        ["-q", "--assert=plain", f"{options.fixture}::{options.test_name}"],
        plugins=[SingleTest()],
    )
    if len(readout.results) == 1:
        print(json.dumps(asdict(readout.results[0])))
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
