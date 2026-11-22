from github_harness_agent.review_mock_replay import (
    _counter_difference,
    counter_caller_site,
)
from github_harness_agent.review_delta_context import stock_delta_context
from github_harness_agent.review_planner import _localized_counter_definition


BASELINE = """def increment(mapping):
    mapping["item_count"] += 1

def original_route(mapping, allowed):
    if allowed:
        before = 1
        increment(
            mapping,
        )
        after = 2
"""
DRAFT = """def increment(mapping):
    mapping["item_count"] += 1

def added_route(mapping, ready):
    if ready:
        before = 5
        increment(
            mapping,
        )
        after = 6

def original_route(mapping, allowed):
    added_route(mapping, allowed)
"""


def _report(failed, callers):
    stack = [["module.py", "increment", 2], *callers]
    return [8, failed, [["a" * 64, [[stack, 0, 1]]]]]


def test_different_counter_callers_include_both_exact_source_windows():
    pinned = _report(None, [["module.py", "original_route", 7]])
    draft = _report(
        0, [["module.py", "added_route", 7], ["module.py", "original_route", 13]]
    )
    maps = ({"module.py": BASELINE}, {"module.py": DRAFT})
    trace = _counter_difference(pinned, draft, "tests/test_demo.py::test_count")
    detail = _counter_difference(pinned, draft, "tests/test_demo.py::test_count", maps)
    assert detail.startswith(trace)
    heading = "\nPinned caller source"
    assert heading not in trace and heading in detail
    excerpts = detail[len(trace) :]
    assert len(excerpts.encode()) <= 1024
    assert "Draft caller source" in excerpts
    assert "if allowed:" in excerpts and "if ready:" in excerpts
    assert "before = 1" in excerpts and "after = 2" in excerpts
    assert "before = 5" in excerpts and "after = 6" in excerpts
    assert excerpts.count("        increment(\n            mapping,\n        )") == 2


def test_counter_caller_source_requires_both_reviewed_files_and_a_common_origin():
    pinned = _report(None, [["module.py", "original_route", 7]])
    draft = _report(0, [["module.py", "added_route", 7]])
    assert "caller source" not in _counter_difference(
        pinned, draft, "tests/test_demo.py::test_count", ({"module.py": BASELINE}, {})
    )
    draft[2][0][1][0][0][0][1] = "other_mutation"
    assert "caller source" not in _counter_difference(
        pinned,
        draft,
        "tests/test_demo.py::test_count",
        ({"module.py": BASELINE}, {"module.py": DRAFT}),
    )


def test_caller_location_uses_existing_enclosing_definition_when_helper_is_new():
    pinned = _report(None, [["module.py", "original_route", 7]])
    draft = _report(
        0, [["module.py", "added_route", 7], ["module.py", "original_route", 13]]
    )
    original, sources = {"module.py": BASELINE}, {"module.py": DRAFT}
    report = _counter_difference(
        pinned, draft, "tests/test_demo.py::test_count", (original, sources)
    )
    _, _, inventory = stock_delta_context(original, sources, "", {}, [])
    localized = _localized_counter_definition(
        original, sources, inventory, counter_caller_site(report)
    )
    assert localized.symbol == ("module.py", "original_route")
    assert "if allowed:" in localized.baseline
    assert "added_route(mapping, allowed)" in localized.current
