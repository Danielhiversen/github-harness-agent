from github_harness_agent.review_mock_replay import (
    _counter_difference,
    compact_observations,
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


def test_skipped_caller_shows_the_pinned_outer_loop_and_guard_only_if_confirmed():
    baseline = (
        BASELINE
        + """
def traverse(mapping, entries):
    for entry in entries:
        if entry:
            original_route(mapping, entry)
"""
    )
    source = (
        BASELINE
        + """
def traverse(mapping, entries):
    for entry in entries:
        increment(mapping)
"""
    )
    pinned_row = (
        baseline.splitlines().index("            original_route(mapping, entry)") + 1
    )
    draft_row = source.splitlines().index("        increment(mapping)") + 1
    pinned = _report(
        None,
        [["module.py", "original_route", 7], ["module.py", "traverse", pinned_row]],
    )
    draft = _report(0, [["module.py", "traverse", draft_row]])
    trace = _counter_difference(pinned, draft, "tests/test_demo.py::test_count")
    detail = _counter_difference(
        pinned,
        draft,
        "tests/test_demo.py::test_count",
        ({"module.py": baseline}, {"module.py": source}),
    )
    assert detail.startswith(trace)
    immediate, outer = detail.split("Pinned outer caller source", 1)
    assert "Pinned caller source" in immediate and "Draft caller source" in immediate
    assert "for entry in entries:" in outer and "if entry:" in outer
    assert "original_route(mapping, entry)" in outer
    assert len(outer.encode()) < 384

    for mocked_baseline, mocked_draft in (
        (
            baseline.replace(
                "original_route(mapping, entry)", "unrelated(mapping, entry)"
            ),
            source,
        ),
        (baseline, source.replace("increment(mapping)\n", "unrelated(mapping)\n")),
    ):
        detail = _counter_difference(
            pinned,
            draft,
            "tests/test_demo.py::test_count",
            ({"module.py": mocked_baseline}, {"module.py": mocked_draft}),
        )
        assert "Pinned caller source" in detail and "Draft caller source" in detail
        assert "Pinned outer caller source" not in detail


def test_counter_trace_accepts_old_events_and_only_bounded_loop_identity_metadata():
    from github_harness_agent.review_counter_trace import valid_report

    paths = ("module.py",)
    report = _report(None, [["module.py", "original_route", 7]])
    assert valid_report(report, paths)
    window = report[2][0]
    window.append([[1, 5, "entry", 1, 1]])
    assert valid_report(report, paths)
    for malformed in (
        [3, 5, "entry", 1, 1],
        [1, 8, "entry", 1, 1],
        [1, 5, "api_key", 1, 1],
        [1, 5, "entry", 100, 1],
        [1, 5, "entry", 1, 0],
        [1, 5, "entry", 1, 1, {"entry": "sensitive"}],
    ):
        window[2] = [malformed]
        assert not valid_report(report, paths)



def test_compact_counter_diagnostic_keeps_the_full_loop_identity_observation():
    reuse = (
        "Draft loop target `" + "entry_" * 7
        + "`: record #1, 3 mutations over 3 loop entries (object identity only). Pinned: 1 observed mutation."
    )
    report = (
        "Auxiliary counterfactual " + "first counterfactual information " * 60
        + "\nAuxiliary mapping counter changes (quoted advisory data). "
        + "\nPinned: 0 -> 1 after module.py:7 " + "baseline frame " * 40
        + "\nDraft: 1 -> 2 after module.py:7 " + "draft frame " * 40
        + "\n" + reuse
        + "\nAuxiliary mock calls " + "mock information " * 60
    )
    compacted = compact_observations(report, 1080)
    assert len(compacted.encode()) <= 1080
    assert reuse in compacted
    assert "Pinned: 0 -> 1 after module.py:7" in compacted
    assert "Draft: 1 -> 2 after module.py:7" in compacted



def test_counter_reuse_can_compare_a_complete_pinned_count_without_loop_bindings():
    from github_harness_agent.review_counter_trace import MAX_EVENTS

    def events(count):
        stack = [["module.py", "increment", 2], ["module.py", "original_route", 7]]
        return [[stack, index, index + 1] for index in range(count)]

    def bindings(count, same_identity=True):
        return [[1, 5, "row", 1 if same_identity else index, index] for index in range(1, count + 1)]

    def detail(pinned, draft, *, lost_window=False):
        preceding = [None] if lost_window else []
        return _counter_difference(
            [8, None, [*preceding, pinned]], [8, len(preceding), [*preceding, draft]],
            "tests/test_demo.py::test_count",
        )

    pinned = ["a" * 64, events(1)]
    draft = ["a" * 64, events(3), bindings(3)]
    observed = detail(pinned, draft)
    assert "3 loop entries" in observed and "Pinned: 1 observed mutation" in observed
    equal_count = events(3)
    equal_count[-1][-1] = 10
    assert "loop target" not in detail(["a" * 64, equal_count], draft)
    assert "loop target" not in detail(pinned, ["a" * 64, events(3), bindings(3, False)])
    assert "loop target" not in detail(pinned, ["a" * 64, events(MAX_EVENTS), bindings(MAX_EVENTS)])
    assert not detail(pinned, draft, lost_window=True)
    assert not detail(pinned, ["b" * 64, events(3), bindings(3)])
