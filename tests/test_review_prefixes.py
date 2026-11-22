from pathlib import Path

from github_harness_agent.review_prefixes import (
    PrefixReadout,
    prefix_improved,
    trace_compound_assert,
)


def test_trace_observes_passed_prefix_without_extra_truthiness_or_skipped_calls(
    tmp_path,
):
    path = tmp_path / "test_fixture.py"
    path.write_text(
        "def test_compound(a, b, c):\n    assert a() and b() and c(), 'failure'\n"
    )
    trace = []

    class Value:
        def __init__(self, value):
            self.value = value

        def __bool__(self):
            trace.append(f"bool {self.value}")
            return self.value

    def a():
        trace.append("a")
        return Value(True)

    def b():
        trace.append("b")
        return Value(False)

    def c():
        raise RuntimeError("skipped clause ran")

    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    readout = trace_compound_assert(path, "test_compound")
    with readout:
        try:
            namespace["test_compound"](a, b, c)
        except AssertionError:
            pass
    assert trace == ["a", "bool True", "b", "bool False"]
    assert readout.results == [
        PrefixReadout(2, ("a()", "b()", "c()"), (True, False, None))
    ]


def test_complex_branching_produces_no_prefix(tmp_path):
    path = tmp_path / "test_fixture.py"
    path.write_text("def test_compound(a, b, c):\n    assert (a or b) and c\n")
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    with trace_compound_assert(path, "test_compound") as readout:
        namespace["test_compound"](True, False, True)
    assert readout.results == []


def test_only_comparable_false_short_circuits_score_as_prefix_progress():
    terms = ("a", "b", "c")
    before = PrefixReadout(4, terms, (False, None, None))
    assert prefix_improved(before, PrefixReadout(4, terms, (True, False, None)))
    assert not prefix_improved(before, PrefixReadout(5, terms, (True, False, None)))
    assert not prefix_improved(before, PrefixReadout(4, terms, (True, None, None)))
    assert not prefix_improved(before, PrefixReadout(4, terms, (True, True, True)))


def test_plain_pytest_replay_keeps_original_failure_and_records_reached_clauses(
    tmp_path,
):
    import subprocess
    import sys

    path = tmp_path / "test_fixture.py"
    path.write_text(
        "def skip_me():\n    raise RuntimeError('never evaluate')\n"
        "def test_compound():\n"
        "    a, b = True, False\n"
        "    assert a and b and skip_me(), 'fixture message'\n"
    )
    command = [
        sys.executable,
        "-m",
        "github_harness_agent.review_prefixes",
        str(path),
        "test_compound",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=Path.cwd(), check=False
    )
    assert result.returncode == 1
    assert "1 failed" in result.stdout
    assert '"values": [true, false, null]' in result.stdout
    assert "fixture message" in result.stdout
    assert "RuntimeError" not in result.stdout
